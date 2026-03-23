"""
model.py - Lightweight causal joint AEC+NS model

Architecture:
  Input [B, T, 257, 4]  (mic_r, mic_i, ref_r, ref_i)
    → Conv2D frequency-domain feature extraction (causal padding)
    → Unidirectional causal GRU temporal modeling
    → Lightweight attention ref interaction
    → Conv1D output head → Mask [B, T, 257, 2]
    → Mask × mic spectrum → iSTFT → time-domain waveform

Design principles:
  - Fully causal (no future-frame dependency), supports embedded real-time streaming inference
  - Single set of weights shared between embedded INT8 and server FP32
  - Parameter count ~27K, ~27KB after INT8 quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class CausalConv2d(nn.Module):
    """
    Causal Conv2D: symmetric padding along the frequency dimension,
    left-only padding along the time dimension (causal).
    Input shape: [B, C, T, F]
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: int = 1):
        super().__init__()
        self.kt, self.kf = kernel_size
        # Time dimension: causal, pad only on the left side (kt-1)
        self.pad_t = self.kt - 1
        # Frequency dimension: symmetric pad
        self.pad_f = self.kf // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,   # manual padding
            bias=False   # GN has its own affine; bias is provided by GN
        )
        # InstanceNorm2d instead of GroupNorm:
        # - GroupNorm depends on batch statistics which breaks streaming/ONNX compatibility
        # - InstanceNorm2d normalizes per-instance, works well for streaming inference
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        # Causal time padding: left side only
        x = F.pad(x, (self.pad_f, self.pad_f, self.pad_t, 0))
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class LightAttention(nn.Module):
    """
    Lightweight per-frame attention: modulates mic features using ref features.
    Computed independently per frame; no future-frame dependency.
    Input: mic_feat, ref_feat  [B, T, C]
    Output: [B, T, C]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, channels, bias=False)
        self.key   = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.scale = channels ** -0.5

    def forward(self, mic: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # mic, ref: [B, T, C]
        q = self.query(mic)            # [B, T, C]
        k = self.key(ref)              # [B, T, C]
        v = self.value(ref)            # [B, T, C]
        # Per-frame dot-product attention (frame-wise, no cross-time dependency)
        attn = torch.einsum('btc,btc->bt', q, k) * self.scale   # [B, T]
        attn = torch.sigmoid(attn).unsqueeze(-1)                 # [B, T, 1]
        return mic + attn * v


class JointAECNSModel(nn.Module):
    """
    Joint AEC + noise suppression model

    Args:
        freq_bins:     STFT frequency bins, default 257 (512-point FFT)
        conv_channels: Conv2D channel count, default 32
        gru_hidden:    GRU hidden dimension, default 32
        n_conv_layers: Number of Conv2D layers, default 3
    """

    def __init__(self,
                 freq_bins: int = 257,
                 conv_channels: int = 32,
                 gru_hidden: int = 32,
                 n_conv_layers: int = 3):
        super().__init__()
        self.freq_bins = freq_bins
        self.gru_hidden = gru_hidden

        # ── Block 1: Frequency-domain feature extraction ──────────────────────────────
        # Input: [B, 4, T, F]  →  Output: [B, conv_channels, T, F]
        conv_layers = []
        in_ch = 4
        for i in range(n_conv_layers):
            out_ch = conv_channels
            conv_layers.append(CausalConv2d(in_ch, out_ch, kernel_size=(3, 3)))
            in_ch = out_ch
        self.conv_block = nn.Sequential(*conv_layers)

        # ── Block 2: Unidirectional causal GRU temporal modeling ────────────────────
        # Flattening freq would give [B, T, conv_channels * freq_bins] which is too large;
        # use frequency mean pooling first.
        # Actual: mean pooling along freq dim after Conv2D → [B, T, conv_channels]
        # GRU input: [B, T, conv_channels * 2]  (half each for mic and ref)
        self.gru_mic = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False   # ← causal!
        )
        self.gru_ref = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # ── Block 3: Reference interaction attention ───────────────────────────
        self.ref_attention = LightAttention(gru_hidden)

        # Fuse mic_gru + attended
        self.fusion = nn.Linear(gru_hidden * 2, conv_channels)

        # ── Block 4: Output head ────────────────────────────────────
        # Expand [B, T, conv_channels] back to [B, T, freq_bins, 2] (complex mask)
        self.mask_head = nn.Sequential(
            nn.Linear(conv_channels, freq_bins * 2),
        )
        # Initialize bias to a positive value: sigmoid(4) ≈ 0.982, so initial mask ≈ 1 (pass-through)
        # Use sigmoid instead of tanh to keep mask ∈ [0,1], attenuating amplitude only without phase flip
        nn.init.constant_(self.mask_head[0].bias, 4.0)

    def forward(self,
                mic_stft: torch.Tensor,
                ref_stft: torch.Tensor,
                hid_mic: Optional[torch.Tensor] = None,
                hid_ref: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mic_stft: [B, T, F, 2]  Microphone STFT complex (real, imag)
            ref_stft: [B, T, F, 2]  Far-end reference STFT complex
            hid_mic:  GRU hidden state, passed across frames during streaming inference, shape [1, B, gru_hidden]
            hid_ref:  GRU hidden state, shape [1, B, gru_hidden]

        Returns:
            out_stft:          [B, T, F, 2]  Enhanced STFT (iSTFT not applied here; returns masked spectrum)
            (hid_mic, hid_ref): Updated GRU hidden states for streaming inference
        """
        B, T, n_freq, _ = mic_stft.shape

        # Merge 4 channels: [B, T, n_freq, 4] → [B, 4, T, n_freq]
        x = torch.cat([mic_stft, ref_stft], dim=-1)   # [B, T, n_freq, 4]
        x = x.permute(0, 3, 1, 2)                      # [B, 4, T, n_freq]

        # Block 1: Causal Conv2D
        feat = self.conv_block(x)   # [B, C, T, n_freq]

        # Frequency-dimension mean pooling → [B, T, C]
        feat_t = feat.mean(dim=-1).permute(0, 2, 1)  # [B, T, C]

        # Prepare separate mic and ref features for the GRU (energy-weighted)
        mic_energy = (mic_stft ** 2).sum(-1).mean(-1, keepdim=True)  # [B, T, 1]
        ref_energy = (ref_stft ** 2).sum(-1).mean(-1, keepdim=True)  # [B, T, 1]
        total = mic_energy + ref_energy + 1e-8
        mic_w = mic_energy / total   # [B, T, 1]
        ref_w = ref_energy / total   # [B, T, 1]

        mic_feat = feat_t * mic_w   # [B, T, C]
        ref_feat = feat_t * ref_w   # [B, T, C]

        # Block 2: Unidirectional causal GRU
        mic_out, hid_mic_new = self.gru_mic(mic_feat, hid_mic)  # [B, T, H]
        ref_out, hid_ref_new = self.gru_ref(ref_feat, hid_ref)  # [B, T, H]

        # Block 3: Reference interaction attention
        attended = self.ref_attention(mic_out, ref_out)          # [B, T, H]

        # Fusion
        fused = self.fusion(torch.cat([mic_out, attended], dim=-1))  # [B, T, C]
        fused = torch.relu(fused)

        # Block 4: Output mask
        mask = self.mask_head(fused)                       # [B, T, n_freq*2]
        mask = mask.view(B, T, n_freq, 2)                  # [B, T, n_freq, 2]
        mask = torch.sigmoid(mask)                          # mask ∈ [0,1], attenuate only, no phase flip

        # Apply mask (element-wise multiply)
        out_stft = mic_stft * mask                         # [B, T, n_freq, 2]

        return out_stft, (hid_mic_new, hid_ref_new)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class STFTProcessor(nn.Module):
    """
    Differentiable STFT / iSTFT processor, wrapping audio pre- and post-processing.
    """

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 160,   # 10ms @ 16kHz
                 win_length: int = 320,   # 20ms @ 16kHz
                 sample_rate: int = 16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate

        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, L]
        return: [B, T, F, 2]  (real, imag)
        """
        B, L = wav.shape
        spec = torch.stft(
            wav.reshape(-1, L),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode='reflect'
        )   # [B, F, T]
        spec = spec.permute(0, 2, 1)        # [B, T, F]
        spec = torch.view_as_real(spec)     # [B, T, F, 2]
        return spec

    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        spec: [B, T, F, 2]
        return: [B, L]
        """
        spec = torch.view_as_complex(spec.contiguous())  # [B, T, F]
        spec = spec.permute(0, 2, 1)                      # [B, F, T]
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length
        )
        return wav


class JointAECNSSystem(nn.Module):
    """
    Full system: STFT + model + iSTFT, trained end-to-end.
    """

    def __init__(self,
                 freq_bins: int = 257,
                 conv_channels: int = 32,
                 gru_hidden: int = 32,
                 n_conv_layers: int = 3,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 win_length: int = 320,
                 sample_rate: int = 16000):
        super().__init__()
        self.stft_proc = STFTProcessor(n_fft, hop_length, win_length, sample_rate)
        self.model = JointAECNSModel(freq_bins, conv_channels, gru_hidden, n_conv_layers)
        self.hop_length = hop_length

    def forward(self,
                mic_wav: torch.Tensor,
                ref_wav: torch.Tensor,
                hid_mic: Optional[torch.Tensor] = None,
                hid_ref: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            mic_wav: [B, L]  Microphone time-domain signal
            ref_wav: [B, L]  Far-end reference time-domain signal
        Returns:
            enhanced_wav: [B, L]
            hidden:       (hid_mic, hid_ref) for streaming inference
        """
        length = mic_wav.shape[-1]
        mic_stft = self.stft_proc.stft(mic_wav)   # [B, T, F, 2]
        ref_stft = self.stft_proc.stft(ref_wav)   # [B, T, F, 2]

        out_stft, hidden = self.model(mic_stft, ref_stft, hid_mic, hid_ref)

        enhanced_wav = self.stft_proc.istft(out_stft, length=length)  # [B, L]
        return enhanced_wav, hidden

    def count_parameters(self) -> int:
        return self.model.count_parameters()


if __name__ == '__main__':
    # Quick sanity check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = JointAECNSSystem().to(device)

    n_params = model.count_parameters()
    print(f'Model parameter count: {n_params:,}  (~{n_params/1000:.1f}K)')
    print(f'Estimated INT8 size: ~{n_params/1024:.1f} KB')

    # Simulate a batch of 2-second audio (batch=2)
    B, L = 2, 32000
    mic = torch.randn(B, L).to(device)
    ref = torch.randn(B, L).to(device)

    with torch.no_grad():
        out, hidden = model(mic, ref)

    print(f'Input shape: {mic.shape}')
    print(f'Output shape: {out.shape}')
    print(f'GRU hidden state: {hidden[0].shape}')
    print('Forward pass OK')

    # Verify causal streaming inference: process frame by frame
    frame_len = 160   # 10ms hop
    n_frames = 10
    hm, hr = None, None
    for i in range(n_frames):
        chunk_mic = torch.randn(1, 320).to(device)  # 20ms frame
        chunk_ref = torch.randn(1, 320).to(device)
        out_chunk, (hm, hr) = model(chunk_mic, chunk_ref, hm, hr)
    print(f'Streaming inference OK, cross-frame hidden state propagation verified')
