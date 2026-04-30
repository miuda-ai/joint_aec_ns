"""
Generate demo audio samples for JointAEC-NS model (streaming mode)
"""
import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from scipy.signal import resample_poly
from scipy.signal import correlate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import JointAECNSModel, STFTProcessor

SR = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320
FREQ_BINS = 257
GRU_HIDDEN = 64


def ensure_mono(wav: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by channel averaging."""
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)


def resample_wav(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample waveform with polyphase filtering."""
    if src_sr == dst_sr:
        return wav.astype(np.float32)
    g = np.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return resample_poly(wav.astype(np.float32), up, down).astype(np.float32)


def shift_wav(wav: np.ndarray, delay_samples: int) -> np.ndarray:
    """Shift waveform in time. Positive delay moves signal later (right-shift)."""
    if delay_samples == 0:
        return wav
    out = np.zeros_like(wav)
    if delay_samples > 0:
        if delay_samples < len(wav):
            out[delay_samples:] = wav[:-delay_samples]
    else:
        k = -delay_samples
        if k < len(wav):
            out[:-k] = wav[k:]
    return out


def estimate_ref_delay_samples(mic: np.ndarray, ref: np.ndarray, sr: int, max_delay_ms: float = 300.0) -> int:
    """
    Estimate delay where mic best correlates with ref.
    Positive result means mic lags ref and ref should be delayed by this amount.
    """
    mic_n = (mic - mic.mean()) / (mic.std() + 1e-8)
    ref_n = (ref - ref.mean()) / (ref.std() + 1e-8)
    max_lag = int(max_delay_ms * sr / 1000.0)

    c = correlate(mic_n, ref_n, mode="full", method="fft")
    lags = np.arange(-len(ref_n) + 1, len(mic_n))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    cc = c[mask]
    ll = lags[mask]
    best = int(ll[np.argmax(np.abs(cc))])
    return best


def pytorch_stream_inference(model, stft_proc, mic_wav, ref_wav):
    """
    Frame-by-frame streaming inference using JointAECNSModel.
    """
    device = "cpu"
    mic_t = torch.from_numpy(mic_wav.astype(np.float32)).unsqueeze(0).to(device)
    ref_t = torch.from_numpy(ref_wav.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        mic_stft = stft_proc.stft(mic_t)  # [1, T, F, 2]
        ref_stft = stft_proc.stft(ref_t)

    T = mic_stft.shape[1]
    hid_mic = None
    hid_ref = None
    outputs = []

    model.eval()
    with torch.no_grad():
        for t in range(T):
            mic_f = mic_stft[:, t:t+1, :, :]
            ref_f = ref_stft[:, t:t+1, :, :]
            out_f, (hid_mic, hid_ref) = model(mic_f, ref_f, hid_mic, hid_ref)
            outputs.append(out_f.squeeze(0).numpy())

    out_stft = np.concatenate(outputs, axis=0)  # [T, F, 2]
    out_stft = torch.from_numpy(out_stft).unsqueeze(0)

    with torch.no_grad():
        enhanced = stft_proc.istft(out_stft, length=mic_wav.shape[0])

    return enhanced.squeeze(0).cpu().numpy()


def onnx_stream_inference(sess, stft_proc, mic_wav, ref_wav, gru_hidden=GRU_HIDDEN):
    """
    Streaming frame-by-frame inference using ONNX model (model_stream.onnx).
    """
    device = "cpu"
    mic_t = torch.from_numpy(mic_wav.astype(np.float32)).unsqueeze(0).to(device)
    ref_t = torch.from_numpy(ref_wav.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        mic_stft = stft_proc.stft(mic_t)  # [1, T, F, 2]
        ref_stft = stft_proc.stft(ref_t)

    mic_stft = mic_stft.numpy()  # [1, T, F, 2]
    ref_stft = ref_stft.numpy()

    T = mic_stft.shape[1]
    hid_mic = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    hid_ref = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    outputs = []

    for t in range(T):
        mic_f = mic_stft[:, t:t+1, :, :]  # [1, 1, F, 2]
        ref_f = ref_stft[:, t:t+1, :, :]

        ort_results = sess.run(
            ["out_stft", "hid_mic_out", "hid_ref_out"],
            {
                "mic_stft": mic_f,
                "ref_stft": ref_f,
                "hid_mic": hid_mic,
                "hid_ref": hid_ref,
            }
        )

        out_f, hid_mic, hid_ref = ort_results
        outputs.append(out_f[0])

    out_stft = np.concatenate(outputs, axis=0)  # [T, F, 2]
    out_stft = torch.from_numpy(out_stft).unsqueeze(0)

    with torch.no_grad():
        enhanced = stft_proc.istft(out_stft, length=mic_wav.shape[0])

    return enhanced.squeeze(0).cpu().numpy()


def _extract_fixed_last_dim(shape, fallback: int) -> int:
    """Extract a fixed last dimension from ONNX shape, otherwise return fallback."""
    if isinstance(shape, (list, tuple)) and len(shape) > 0:
        last = shape[-1]
        if isinstance(last, int) and last > 0:
            return last
        if isinstance(last, str) and last.isdigit():
            return int(last)
    return fallback


def infer_onnx_gru_hidden(sess: ort.InferenceSession, fallback: int) -> int:
    """Infer GRU hidden dim from ONNX input signature (hid_mic/hid_ref)."""
    hidden = fallback
    for inp in sess.get_inputs():
        if inp.name in ("hid_mic", "hid_ref"):
            hidden = _extract_fixed_last_dim(inp.shape, hidden)
            break
    return hidden


def main():
    parser = argparse.ArgumentParser(description="Generate demo audio samples (streaming mode)")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Use ONNX model for inference (specify path to .onnx file)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth",
                        help="PyTorch checkpoint path")
    parser.add_argument("--hparams", type=str, default="checkpoints/hparams.json",
                        help="Hyperparameters JSON path")
    parser.add_argument("--conv_channels", type=int, default=None,
                        help="override conv_channels")
    parser.add_argument("--gru_hidden", type=int, default=None,
                        help="override gru_hidden")
    parser.add_argument("--mic", type=str, default="mix_capture_echo.wav",
                        help="microphone capture wav path")
    parser.add_argument("--ref", type=str, default="mix_render_play.wav",
                        help="render/playback reference wav path")
    parser.add_argument("--clean", type=str, default="mix_near_answer.wav",
                        help="optional clean reference wav path for alignment")
    parser.add_argument("--out", type=str, default=None,
                        help="output enhanced wav path")
    parser.add_argument("--model_sr", type=int, default=SR,
                        help="model processing sample rate")
    parser.add_argument("--out_sr", type=str, default="input", choices=["input", "model"],
                        help="output wav sample rate: input (default) or model")
    parser.add_argument("--ref_delay_ms", type=float, default=0.0,
                        help="manual reference delay in ms (positive delays ref)")
    parser.add_argument("--auto_ref_delay", action="store_true",
                        help="estimate mic/ref delay by cross-correlation and align reference")
    parser.add_argument("--max_delay_ms", type=float, default=300.0,
                        help="max absolute delay (ms) for auto delay estimation")
    args = parser.parse_args()

    use_onnx = args.onnx is not None

    # Load hparams for both ONNX and PyTorch modes
    gru_hidden = GRU_HIDDEN
    if os.path.exists(args.hparams):
        with open(args.hparams) as f:
            hp = json.load(f)
        gru_hidden = hp.get("gru_hidden", GRU_HIDDEN)
    if args.gru_hidden is not None:
        gru_hidden = args.gru_hidden

    # Create STFTProcessor
    stft_proc = STFTProcessor(N_FFT, HOP_LENGTH, WIN_LENGTH, args.model_sr)

    if use_onnx:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            args.onnx,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )
        inferred_hidden = infer_onnx_gru_hidden(sess, gru_hidden)
        if inferred_hidden != gru_hidden:
            print(f"ONNX hidden dim inferred as {inferred_hidden} (override from {gru_hidden})")
            gru_hidden = inferred_hidden
        print(f"Using ONNX model: {args.onnx}, gru_hidden={gru_hidden}")
        print("ONNX model loaded")
        out_dir = "demo_onnx/"
    else:
        print("Loading PyTorch model (streaming mode)...")
        if os.path.exists(args.hparams):
            with open(args.hparams) as f:
                hp = json.load(f)
            freq_bins = hp.get("freq_bins", 257)
            conv_channels = hp.get("conv_channels", 64)
            gru_hidden = hp.get("gru_hidden", 64)
            n_conv_layers = hp.get("n_conv_layers", 3)
        else:
            freq_bins, conv_channels, gru_hidden, n_conv_layers = 257, 64, 64, 3

        # Override with command line arguments if provided
        if args.conv_channels is not None:
            conv_channels = args.conv_channels
        if args.gru_hidden is not None:
            gru_hidden = args.gru_hidden

        ckpt = torch.load(args.ckpt, map_location="cpu")
        # Use JointAECNSModel for streaming inference
        model = JointAECNSModel(
            freq_bins=freq_bins, conv_channels=conv_channels,
            gru_hidden=gru_hidden, n_conv_layers=n_conv_layers,
        ).to("cpu")
        state = ckpt.get("model_state", ckpt)
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        model_keys = set(model.state_dict().keys())
        filtered_state = {k: v for k, v in new_state.items() if k in model_keys}
        model.load_state_dict(filtered_state)
        model.eval()
        print(f"PyTorch model loaded: epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f}")
        out_dir = "demo/"

    os.makedirs(out_dir, exist_ok=True)

    mic_path = args.mic
    ref_path = args.ref
    clean_path = args.clean

    mic_wav, mic_sr = sf.read(mic_path)
    ref_wav, ref_sr = sf.read(ref_path)
    mic_wav = ensure_mono(np.asarray(mic_wav))
    ref_wav = ensure_mono(np.asarray(ref_wav))

    if ref_sr != mic_sr:
        ref_wav = resample_wav(ref_wav, ref_sr, mic_sr)

    if os.path.exists(clean_path):
        clean_wav, clean_sr = sf.read(clean_path)
        clean_wav = ensure_mono(np.asarray(clean_wav))
        if clean_sr != mic_sr:
            clean_wav = resample_wav(clean_wav, clean_sr, mic_sr)
        L = min(len(mic_wav), len(ref_wav), len(clean_wav))
    else:
        print(f"Warning: clean reference not found: {clean_path}. Proceeding without it.")
        L = min(len(mic_wav), len(ref_wav))
    mic_wav, ref_wav = mic_wav[:L], ref_wav[:L]

    if args.auto_ref_delay:
        est_delay = estimate_ref_delay_samples(mic_wav, ref_wav, mic_sr, max_delay_ms=args.max_delay_ms)
        ref_wav = shift_wav(ref_wav, est_delay)
        print(f"Auto ref delay applied: {est_delay} samples ({est_delay * 1000.0 / mic_sr:.2f} ms)")
    elif args.ref_delay_ms != 0.0:
        delay_samples = int(round(args.ref_delay_ms * mic_sr / 1000.0))
        ref_wav = shift_wav(ref_wav, delay_samples)
        print(f"Manual ref delay applied: {delay_samples} samples ({args.ref_delay_ms:.2f} ms)")

    if mic_sr != args.model_sr:
        mic_wav = resample_wav(mic_wav, mic_sr, args.model_sr)
        ref_wav = resample_wav(ref_wav, mic_sr, args.model_sr)
        print(f"Resampled input audio: {mic_sr} -> {args.model_sr} Hz for model inference")

    if use_onnx:
        enhanced = onnx_stream_inference(sess, stft_proc, mic_wav, ref_wav, gru_hidden=gru_hidden)
    else:
        enhanced = pytorch_stream_inference(model, stft_proc, mic_wav, ref_wav)

    write_sr = mic_sr if args.out_sr == "input" else args.model_sr
    if write_sr != args.model_sr:
        enhanced = resample_wav(enhanced, args.model_sr, write_sr)
        print(f"Resampled output audio: {args.model_sr} -> {write_sr} Hz for saving")

    out_path = args.out if args.out is not None else f"{out_dir}/enhanced.wav"
    sf.write(out_path, enhanced, write_sr)
    print(f"Saved enhanced wav: {out_path}")
    
if __name__ == "__main__":
    main()
