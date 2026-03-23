"""
losses.py - Joint loss function

Combination:
  L_total = α * L_sisnr + β * L_freq

  L_sisnr: Time-domain SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
  L_freq:  Frequency-domain complex MSE (real + imaginary parts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func_nn
from typing import Tuple


def si_snr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SNR Loss (lower is better; returns negative SI-SNR)

    Args:
        pred:   [B, L]  Enhanced time-domain waveform
        target: [B, L]  Clean target speech
        eps:    Numerical stability term

    Returns:
        loss: Scalar, batch-averaged -SI-SNR (dB)
    """
    # Remove mean (cast to FP32 to avoid precision issues with FP16/BF16)
    pred   = pred.float()   - pred.float().mean(dim=-1, keepdim=True)
    target = target.float() - target.float().mean(dim=-1, keepdim=True)

    # Projection
    dot = (pred * target).sum(dim=-1, keepdim=True)         # [B, 1]
    s_target_norm = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot / s_target_norm * target                      # [B, L]

    # Noise component
    noise = pred - proj

    # SI-SNR: gradient-stable version
    proj_pow  = (proj  ** 2).sum(dim=-1)                    # [B]  ≥ 0
    noise_pow = (noise ** 2).sum(dim=-1)                    # [B]  ≥ 0

    # Lower bound of noise_pow = proj_pow / 10^(SNR_MAX/10)
    # Equivalent to capping SI-SNR at SNR_MAX dB, keeping gradients bounded
    snr_max_linear = 10 ** (35.0 / 10)                     # 3162.3
    noise_floor = proj_pow.detach() / snr_max_linear + eps  # Adaptive w.r.t. proj; detach keeps proj gradient unaffected
    noise_pow = torch.clamp(noise_pow, min=noise_floor)

    si_snr = 10 * torch.log10((proj_pow + eps) / noise_pow) # [B]

    # Lower bound -100 dB (the original -35 dB caused 75% of batch gradients to be 0
    # at the start of training, preventing model updates).
    # Relaxing to -100 dB ensures valid gradients even when model output is near zero.
    si_snr = si_snr.clamp(min=-100.0)

    return -si_snr.mean()


def freq_mse_loss(pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    """
    Frequency-domain complex MSE Loss

    Args:
        pred_stft:   [B, T, F, 2]  Predicted STFT (real + imaginary parts)
        target_stft: [B, T, F, 2]  Target STFT

    Returns:
        loss: Scalar
    """
    return func_nn.mse_loss(pred_stft, target_stft)


def mag_loss(pred_stft: torch.Tensor, target_stft: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Spectral magnitude L1 Loss (phase-insensitive; helps stabilise early training)

    Args:
        pred_stft:   [B, T, F, 2]
        target_stft: [B, T, F, 2]

    Returns:
        loss: Scalar
    """
    # Use sqrt(r²+i²+eps) rather than sqrt(r²+i²)+eps.
    # The latter has gradient = 1/(2·sqrt(0)) = inf when r²+i²→0;
    # the former has gradient = x/sqrt(x+eps) → 0 (bounded).
    pred_mag   = (pred_stft[..., 0] ** 2 + pred_stft[..., 1] ** 2 + eps).sqrt()   # [B, T, F]
    target_mag = (target_stft[..., 0] ** 2 + target_stft[..., 1] ** 2 + eps).sqrt()
    return func_nn.l1_loss(pred_mag, target_mag)


class JointLoss(nn.Module):
    """
    Joint loss: SI-SNR + frequency-domain MSE + magnitude L1

    Args:
        alpha: SI-SNR weight, default 0.5
        beta:  Frequency-domain complex MSE weight, default 0.3
        gamma: Magnitude L1 weight, default 0.2
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self,
                pred_wav: torch.Tensor,
                target_wav: torch.Tensor,
                pred_stft: torch.Tensor,
                target_stft: torch.Tensor
                ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_wav:    [B, L]       Enhanced time-domain signal
            target_wav:  [B, L]       Clean target signal
            pred_stft:   [B, T, F, 2] Enhanced STFT
            target_stft: [B, T, F, 2] Target STFT

        Returns:
            total_loss: Scalar
            loss_dict:  Per-component losses (for TensorBoard logging)
        """
        l_sisnr = si_snr_loss(pred_wav, target_wav)
        l_freq  = freq_mse_loss(pred_stft, target_stft)
        l_mag   = mag_loss(pred_stft, target_stft)

        total = self.alpha * l_sisnr + self.beta * l_freq + self.gamma * l_mag

        loss_dict = {
            'loss_total': total.item(),
            'loss_sisnr': l_sisnr.item(),
            'loss_freq':  l_freq.item(),
            'loss_mag':   l_mag.item(),
        }
        return total, loss_dict


if __name__ == '__main__':
    B, L, T, F = 4, 32000, 200, 257
    pred_wav    = torch.randn(B, L)
    target_wav  = torch.randn(B, L)
    pred_stft   = torch.randn(B, T, F, 2)
    target_stft = torch.randn(B, T, F, 2)

    criterion = JointLoss()
    loss, info = criterion(pred_wav, target_wav, pred_stft, target_stft)
    print(f'Total loss: {loss.item():.4f}')
    for k, v in info.items():
        print(f'  {k}: {v:.4f}')
    print('losses.py OK')
