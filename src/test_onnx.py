"""
test_onnx.py - Streaming inference test using ONNX Runtime

Features:
  1. Load export/model_stream.onnx
  2. Perform streaming frame-level inference on real audio (results/compare/sample_029997/mic_noisy.wav)
  3. Compare with PyTorch model output, print max error
  4. Save output as export/onnx_output.wav
  5. Measure single-threaded CPU inference latency (ms/frame)

Usage:
  python3 src/test_onnx.py
  python3 src/test_onnx.py --ckpt checkpoints/best.pth
                           --audio results/compare/sample_029997/mic_noisy.wav
                           --ref   results/compare/sample_029997/ref.wav
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import soundfile as sf
import scipy.signal
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import JointAECNSModel

SR = 16000
N_FFT = 512
HOP_LENGTH = 160     # 10ms @ 16kHz
WIN_LENGTH = 320     # 20ms Hann window
FREQ_BINS = 257      # N_FFT/2 + 1


# STFT utility (numpy/scipy)

def numpy_stft(wav: np.ndarray,
               n_fft: int = N_FFT,
               hop_length: int = HOP_LENGTH,
               win_length: int = WIN_LENGTH) -> np.ndarray:
    """
    Compute STFT, returns [T, F, 2] (real, imag).
    Uses scipy.signal.stft, aligned with PyTorch's torch.stft.
    """
    window = scipy.signal.get_window("hann", win_length, fftbins=True).astype(np.float32)
    # scipy stft: nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft
    _, _, Zxx = scipy.signal.stft(
        wav.astype(np.float32),
        fs=SR,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    # Zxx: [F, T] complex
    Zxx = Zxx.T                            # [T, F]
    stft = np.stack([Zxx.real, Zxx.imag], axis=-1)  # [T, F, 2]
    return stft.astype(np.float32)


def numpy_istft(stft: np.ndarray,
                n_fft: int = N_FFT,
                hop_length: int = HOP_LENGTH,
                win_length: int = WIN_LENGTH,
                length: int = None) -> np.ndarray:
    """
    Reconstruct time-domain waveform from [T, F, 2] (real, imag).
    """
    window = scipy.signal.get_window("hann", win_length, fftbins=True).astype(np.float32)
    Zxx = stft[..., 0] + 1j * stft[..., 1]   # [T, F]
    Zxx = Zxx.T                                # [F, T]
    _, wav = scipy.signal.istft(
        Zxx,
        fs=SR,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=True,
    )
    wav = wav.astype(np.float32)
    if length is not None:
        if len(wav) >= length:
            wav = wav[:length]
        else:
            wav = np.pad(wav, (0, length - len(wav)))
    return wav


# PyTorch inference (reference)

def pytorch_inference(model: JointAECNSModel,
                      mic_stft: np.ndarray,
                      ref_stft: np.ndarray,
                      gru_hidden: int = 64) -> np.ndarray:
    """
    Frame-by-frame inference using PyTorch model (for ONNX comparison).
    mic_stft, ref_stft: [T, F, 2]
    Returns: out_stft [T, F, 2]
    """
    T = mic_stft.shape[0]
    hid_mic = None
    hid_ref = None
    outputs = []

    model.eval()
    with torch.no_grad():
        for t in range(T):
            mic_f = torch.from_numpy(mic_stft[t:t+1]).unsqueeze(0)  # [1, 1, F, 2]
            ref_f = torch.from_numpy(ref_stft[t:t+1]).unsqueeze(0)  # [1, 1, F, 2]
            out_f, (hid_mic, hid_ref) = model(mic_f, ref_f, hid_mic, hid_ref)
            outputs.append(out_f.squeeze(0).numpy())  # [1, F, 2]

    return np.concatenate(outputs, axis=0)  # [T, F, 2]


# ONNX streaming inference

def onnx_stream_inference(sess: ort.InferenceSession,
                          mic_stft: np.ndarray,
                          ref_stft: np.ndarray,
                          gru_hidden: int = 64) -> tuple:
    """
    Frame-by-frame inference using ONNX model.
    mic_stft, ref_stft: [T, F, 2]
    Returns: (out_stft [T, F, 2], times_ms list)
    """
    T = mic_stft.shape[0]
    hid_mic = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    hid_ref = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    outputs = []
    times_ms = []

    for t in range(T):
        mic_f = mic_stft[t:t+1][np.newaxis]  # [1, 1, F, 2]
        ref_f = ref_stft[t:t+1][np.newaxis]  # [1, 1, F, 2]

        t0 = time.perf_counter()
        ort_results = sess.run(
            ["out_stft", "hid_mic_out", "hid_ref_out"],
            {
                "mic_stft": mic_f,
                "ref_stft": ref_f,
                "hid_mic":  hid_mic,
                "hid_ref":  hid_ref,
            }
        )
        t1 = time.perf_counter()

        out_f, hid_mic, hid_ref = ort_results
        outputs.append(out_f[0])  # [1, F, 2]
        times_ms.append((t1 - t0) * 1000.0)

    return np.concatenate(outputs, axis=0), times_ms  # [T, F, 2]


# Main function

def parse_args():
    p = argparse.ArgumentParser(description="ONNX streaming inference test")
    p.add_argument("--onnx",    type=str,
                   default="export/model_stream.onnx",
                   help="ONNX model path")
    p.add_argument("--ckpt",    type=str,
                   default="checkpoints/best.pth",
                   help="PyTorch checkpoint (for comparison)")
    p.add_argument("--hparams", type=str,
                   default="checkpoints/hparams.json",
                   help="hparams.json path")
    p.add_argument("--audio",   type=str,
                   default="results/compare/sample_029997/mic_noisy.wav",
                   help="Test audio path")
    p.add_argument("--ref",     type=str,
                   default="results/compare/sample_029997/ref.wav",
                   help="Reference audio path")
    p.add_argument("--out",     type=str,
                   default="export/onnx_output.wav",
                   help="ONNX output wav save path")
    p.add_argument("--warmup",  type=int, default=10,
                   help="Number of warmup frames (not counted in latency)")
    return p.parse_args()


def main():
    args = parse_args()

    # Read hyperparameters
    if os.path.exists(args.hparams):
        with open(args.hparams) as f:
            hp = json.load(f)
        freq_bins     = hp.get("freq_bins", 257)
        conv_channels = hp.get("conv_channels", 64)
        gru_hidden    = hp.get("gru_hidden", 64)
        n_conv_layers = hp.get("n_conv_layers", 3)
    else:
        freq_bins, conv_channels, gru_hidden, n_conv_layers = 257, 64, 64, 3

    # Load audio
    print(f"Loading audio: {args.audio}")
    if not os.path.exists(args.audio):
        # Generate synthetic test signal if audio file doesn't exist
        print("  Audio file not found, generating synthetic test signal...")
        duration = 3.0  # 3 seconds
        L = int(SR * duration)
        # Generate a simple sine wave as test signal
        t = np.linspace(0, duration, L)
        freq = 440  # A4 note
        mic_wav = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        ref_wav = mic_wav.copy()  # Reference can be same as mic for simplest test
        print(f"  Generated synthetic signal: {duration}s, {L} samples")
    else:
        mic_wav, sr = sf.read(args.audio, dtype="float32")
        assert sr == SR, f"Sample rate mismatch: {sr} != {SR}"

        if os.path.exists(args.ref):
            ref_wav, _ = sf.read(args.ref, dtype="float32")
        else:
            print("  Reference audio not found, using zero signal as ref")
            ref_wav = np.zeros_like(mic_wav)

    # Align length
    L = min(len(mic_wav), len(ref_wav))
    mic_wav, ref_wav = mic_wav[:L], ref_wav[:L]
    duration_s = L / SR
    print(f"Audio duration: {duration_s:.2f}s  ({L} samples)")

    # Compute STFT
    print("\nComputing STFT (numpy/scipy)...")
    mic_stft = numpy_stft(mic_wav)   # [T, F, 2]
    ref_stft = numpy_stft(ref_wav)   # [T, F, 2]
    T = mic_stft.shape[0]
    print(f"  STFT shape: {mic_stft.shape}  (T={T} frames)")

    # Load ONNX model
    print(f"\nLoading ONNX model: {args.onnx}")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1   # single-threaded
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        args.onnx,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    print(f"  Loaded successfully")

    # Warmup
    print(f"\nWarming up {args.warmup} frames...")
    hid_mic_w = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    hid_ref_w = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    for t in range(min(args.warmup, T)):
        mic_f = mic_stft[t:t+1][np.newaxis]
        ref_f = ref_stft[t:t+1][np.newaxis]
        ort_out = sess.run(
            ["out_stft", "hid_mic_out", "hid_ref_out"],
            {"mic_stft": mic_f, "ref_stft": ref_f,
             "hid_mic": hid_mic_w, "hid_ref": hid_ref_w}
        )
        _, hid_mic_w, hid_ref_w = ort_out

    # ONNX streaming inference (timing)
    print(f"\nONNX streaming inference ({T} frames)...")
    onnx_stft, times_ms = onnx_stream_inference(
        sess, mic_stft, ref_stft, gru_hidden
    )

    times_arr = np.array(times_ms)
    mean_ms = times_arr.mean()
    std_ms  = times_arr.std()
    rtf     = (times_arr.sum() / 1000.0) / duration_s
    print(f"  ONNX latency stats ({T} frames):")
    print(f"    mean = {mean_ms:.3f} ms/frame  std = {std_ms:.3f} ms")
    print(f"    min  = {times_arr.min():.3f} ms  max = {times_arr.max():.3f} ms")
    print(f"    RTF  = {rtf:.4f}  (processing time / audio duration, <1 means real-time feasible)")

    # iSTFT -> time domain
    print("\nReconstructing time-domain waveform (iSTFT)...")
    enhanced_wav = numpy_istft(onnx_stft, length=L)
    print(f"  Enhanced waveform length: {len(enhanced_wav)} samples")

    # Save output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, enhanced_wav, SR)
    print(f"  Saved to: {args.out}")

    # Compare with PyTorch (if checkpoint exists)
    has_pytorch = os.path.exists(args.ckpt)
    if has_pytorch:
        print("\nLoading PyTorch model for comparison...")
        pt_model = JointAECNSModel(
            freq_bins=freq_bins,
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            n_conv_layers=n_conv_layers,
        )
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        model_keys = set(pt_model.state_dict().keys())
        filtered_state = {k: v for k, v in new_state.items() if k in model_keys}
        pt_model.load_state_dict(filtered_state)
        pt_model.eval()

        print("  PyTorch frame-by-frame inference (for comparison)...")
        pt_stft = pytorch_inference(pt_model, mic_stft, ref_stft, gru_hidden)

        # Compare spectral domain error
        max_diff  = np.abs(onnx_stft - pt_stft).max()
        mean_diff = np.abs(onnx_stft - pt_stft).mean()
        print(f"\n  ONNX vs PyTorch spectral domain comparison:")
        print(f"    Max error (max diff) : {max_diff:.2e}")
        print(f"    Mean error (mean diff): {mean_diff:.2e}")
        if max_diff < 1e-4:
            print(f"    Consistency verified (max diff < 1e-4)")
        else:
            print(f"    WARNING: Large error, please check (max diff = {max_diff:.2e})")

        # Compare time domain error (after iSTFT)
        pt_wav = numpy_istft(pt_stft, length=L)
        wav_diff = np.abs(enhanced_wav - pt_wav).max()
        print(f"\n  ONNX vs PyTorch time domain comparison (after iSTFT):")
        print(f"    Max error (max diff): {wav_diff:.2e}")
    else:
        print("\n  PyTorch checkpoint not found, skipping PyTorch comparison")
        max_diff = 0.0

    print("\n" + "="*55)
    print("Test Summary")
    print("="*55)
    print(f"  Audio: {args.audio}")
    print(f"  Duration: {duration_s:.2f}s  Frames: {T}")
    print(f"  ONNX latency: {mean_ms:.3f} +/- {std_ms:.3f} ms/frame")
    print(f"  RTF: {rtf:.4f}")
    if has_pytorch:
        print(f"  Spectral max error: {max_diff:.2e}")
    print(f"  Output file: {args.out}")
    print("="*55)


if __name__ == "__main__":
    main()
