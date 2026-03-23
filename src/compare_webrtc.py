"""
compare_webrtc.py - Compare our model vs WebRTC NS / WebRTC AEC+NS / RNNoise baselines

Usage:
  # After training completes
  python3 src/compare_webrtc.py --ckpt checkpoints/best.pth --n_samples 50

  # Specify output directory (saves comparison wavs)
  python3 src/compare_webrtc.py --ckpt checkpoints/best.pth --out_dir results/compare

Output:
  - Terminal: PESQ / STOI / SI-SNR / ERLE comparison table
  - results/compare/sample_<idx>/  contains:
      mic_noisy.wav       Microphone input (echo + noise)
      ref.wav             Far-end reference signal
      clean_target.wav    Target clean speech
      webrtc_ns.wav       WebRTC NS output
      webrtc_aec_ns.wav   SpeexDSP AEC + WebRTC NS output
      rnnoise.wav         RNNoise output
      rnnoise_aec.wav     SpeexDSP AEC + RNNoise output
      our_model.wav       Our model output

Dependencies:
  pip install pesq pystoi webrtc-noise-gain speexdsp pyrnnoise
"""

import os
import sys
import argparse
import random
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import JointAECNSSystem

# ── Optional dependencies (graceful degradation) ─────────────
try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("[Warning] pesq not installed, skipping PESQ metric. pip install pesq")

try:
    from pystoi import stoi as stoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("[Warning] pystoi not installed, skipping STOI metric. pip install pystoi")

try:
    from webrtc_noise_gain import AudioProcessor as WebRTCAP
    HAS_WEBRTC_NS = True
except ImportError:
    HAS_WEBRTC_NS = False
    print("[Warning] webrtc_noise_gain not installed, skipping WebRTC NS. pip install webrtc-noise-gain")

try:
    from speexdsp import EchoCanceller
    HAS_SPEEX_AEC = True
except ImportError:
    HAS_SPEEX_AEC = False
    print("[Warning] speexdsp not installed, skipping WebRTC AEC+NS. pip install speexdsp")

try:
    from pyrnnoise import RNNoise
    HAS_RNNOISE = True
except ImportError:
    HAS_RNNOISE = False
    print("[Warning] pyrnnoise not installed, skipping RNNoise. pip install pyrnnoise")

SR = 16000
FRAME = 160   # 10ms @ 16kHz


# ─────────────────────────── Metrics ─────────────────────────

def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est * ref).sum() / ((ref * ref).sum() + 1e-8)
    proj  = alpha * ref
    noise = est - proj
    return float(10 * np.log10((proj ** 2).sum() / ((noise ** 2).sum() + 1e-8)))


def erle(mic: np.ndarray, enhanced: np.ndarray) -> float:
    """Echo Return Loss Enhancement: higher means more echo suppressed"""
    return float(10 * np.log10(
        (np.mean(mic ** 2) + 1e-10) / (np.mean(enhanced ** 2) + 1e-10)
    ))


def calc_metrics(est: np.ndarray, ref_clean: np.ndarray,
                 mic_noisy: np.ndarray) -> dict:
    L = min(len(est), len(ref_clean), len(mic_noisy))
    est, ref_clean, mic_noisy = est[:L], ref_clean[:L], mic_noisy[:L]

    m = {
        'si_snr':  si_snr(est, ref_clean),
        'erle':    erle(mic_noisy, est),
    }
    if HAS_PESQ:
        try:
            m['pesq'] = float(pesq_fn(SR, ref_clean, est, 'wb'))
        except Exception:
            m['pesq'] = float('nan')
    if HAS_STOI:
        try:
            m['stoi'] = float(stoi_fn(ref_clean, est, SR, extended=False))
        except Exception:
            m['stoi'] = float('nan')
    return m


# ─────────────────────────── WebRTC NS ───────────────────────

def webrtc_ns(wav: np.ndarray) -> np.ndarray:
    """WebRTC noise suppression (NS only, no AEC)"""
    if not HAS_WEBRTC_NS:
        return wav.copy()

    ap  = WebRTCAP(SR, 0)   # (sample_rate, agc_target_dbfs=0 means no gain)
    wav_i16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    out_i16 = np.zeros_like(wav_i16)

    for start in range(0, len(wav_i16) - FRAME + 1, FRAME):
        chunk  = wav_i16[start: start + FRAME].tobytes()
        result = ap.Process10ms(chunk)
        out_i16[start: start + FRAME] = np.frombuffer(result.audio, dtype=np.int16)

    return out_i16.astype(np.float32) / 32767.0


# ─────────────────────────── WebRTC AEC + NS ─────────────────

def webrtc_aec_ns(mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """SpeexDSP AEC → WebRTC NS cascade"""
    if not HAS_SPEEX_AEC:
        return webrtc_ns(mic)

    # Step 1: SpeexDSP AEC
    ec = EchoCanceller.create(FRAME, 2048, SR)
    mic_i16 = (np.clip(mic, -1.0, 1.0) * 32767).astype(np.int16)
    ref_i16 = (np.clip(ref, -1.0, 1.0) * 32767).astype(np.int16)
    aec_i16 = np.zeros_like(mic_i16)

    for start in range(0, len(mic_i16) - FRAME + 1, FRAME):
        mic_chunk = mic_i16[start: start + FRAME].tobytes()
        ref_chunk = ref_i16[start: start + FRAME].tobytes()
        out_bytes = ec.process(mic_chunk, ref_chunk)
        aec_i16[start: start + FRAME] = np.frombuffer(out_bytes, dtype=np.int16)

    aec_f32 = aec_i16.astype(np.float32) / 32767.0

    # Step 2: WebRTC NS
    return webrtc_ns(aec_f32)


# ─────────────────────────── RNNoise ─────────────────────────

RNN_FRAME = 480   # RNNoise fixed frame: 30ms @ 16kHz

def rnnoise_denoise(wav: np.ndarray) -> np.ndarray:
    """RNNoise noise suppression (no AEC)"""
    if not HAS_RNNOISE:
        return wav.copy()

    rnn = RNNoise(SR)
    # denoise_chunk returns list of (speech_prob, frame_i16 shape=(1,480))
    frames = list(rnn.denoise_chunk(wav.astype(np.float32), partial=True))
    if not frames:
        return wav.copy()
    out_i16 = np.concatenate([f[1].squeeze() for f in frames], axis=0)
    # Trim/pad to original length
    out = out_i16.astype(np.float32) / 32767.0
    L = len(wav)
    if len(out) >= L:
        return out[:L]
    return np.pad(out, (0, L - len(out)))


def rnnoise_aec(mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """SpeexDSP AEC → RNNoise cascade"""
    if not HAS_SPEEX_AEC:
        return rnnoise_denoise(mic)

    # Step 1: SpeexDSP AEC
    ec = EchoCanceller.create(FRAME, 2048, SR)
    mic_i16 = (np.clip(mic, -1.0, 1.0) * 32767).astype(np.int16)
    ref_i16 = (np.clip(ref, -1.0, 1.0) * 32767).astype(np.int16)
    aec_i16 = np.zeros_like(mic_i16)

    for start in range(0, len(mic_i16) - FRAME + 1, FRAME):
        mic_chunk = mic_i16[start: start + FRAME].tobytes()
        ref_chunk = ref_i16[start: start + FRAME].tobytes()
        out_bytes = ec.process(mic_chunk, ref_chunk)
        aec_i16[start: start + FRAME] = np.frombuffer(out_bytes, dtype=np.int16)

    aec_f32 = aec_i16.astype(np.float32) / 32767.0

    # Step 2: RNNoise
    return rnnoise_denoise(aec_f32)


# ─────────────────────────── Our model ───────────────────────

def our_model_enhance(model: JointAECNSSystem,
                      mic: np.ndarray,
                      ref: np.ndarray,
                      device: str) -> np.ndarray:
    """
    Streaming frame-by-frame inference using JointAECNSModel.
    Required because InstanceNorm2d in the model causes different behavior
    between batch processing (entire audio at once) and streaming (frame-by-frame).
    """
    # Move entire model to CPU for consistent computation
    model = model.to('cpu')

    mic_t = torch.from_numpy(mic.astype(np.float32)).unsqueeze(0)
    ref_t = torch.from_numpy(ref.astype(np.float32)).unsqueeze(0)

    stft_proc = model.stft_proc

    with torch.no_grad():
        mic_stft = stft_proc.stft(mic_t)
        ref_stft = stft_proc.stft(ref_t)

    T = mic_stft.shape[1]
    hid_mic, hid_ref = None, None
    outputs = []

    model.model.eval()
    with torch.no_grad():
        for t in range(T):
            out_f, (hid_mic, hid_ref) = model.model(
                mic_stft[:, t:t+1, :, :],
                ref_stft[:, t:t+1, :, :],
                hid_mic, hid_ref
            )
            outputs.append(out_f.squeeze(0).numpy())

    out_stft = torch.from_numpy(np.concatenate(outputs, axis=0)).unsqueeze(0)

    with torch.no_grad():
        enhanced = stft_proc.istft(out_stft, length=mic.shape[0])

    return enhanced.squeeze(0).numpy()


# ─────────────────────────── Main ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Model vs WebRTC NS+AEC comparison evaluation')
    p.add_argument('--ckpt',       type=str, required=True,
                   help='Model checkpoint path (best.pth)')
    p.add_argument('--sim_dir',    type=str, default='data/simulated',
                   help='Simulated data directory')
    p.add_argument('--out_dir',    type=str, default='results/compare',
                   help='Output directory (saves comparison wavs and metrics csv)')
    p.add_argument('--n_samples',  type=int, default=50,
                   help='Number of samples to evaluate (taken from the end of the validation set)')
    p.add_argument('--save_wav',   action='store_true', default=True,
                   help='Save per-method output wavs')
    p.add_argument('--no_save_wav',action='store_true', default=False,
                   help='Skip saving wavs, print metrics only')
    p.add_argument('--conv_channels', type=int, default=48)
    p.add_argument('--gru_hidden',    type=int, default=48)
    p.add_argument('--n_fft',         type=int, default=512)
    p.add_argument('--hop_length',    type=int, default=160)
    p.add_argument('--win_length',    type=int, default=320)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_wav = args.save_wav and not args.no_save_wav
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    ckpt = torch.load(args.ckpt, map_location=device)
    # checkpoint contains JointAECNSModel state_dict (no "model." prefix, no stft_proc)
    # but JointAECNSSystem wraps it as model.xxx with stft_proc
    state = ckpt.get('model_state', ckpt)
    new_state = {}
    for k, v in state.items():
        # Add "model." prefix to match JointAECNSSystem structure
        new_state['model.' + k] = v

    model = JointAECNSSystem(
        conv_channels=args.conv_channels,
        gru_hidden=args.gru_hidden,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    ).to(device)
    # Filter out stft_proc.window (not in checkpoint) and load with strict=False
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in new_state.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    print(f'Parameters: {model.count_parameters():,}')
    print(f'Checkpoint: epoch={ckpt["epoch"]}  val_loss={ckpt["val_loss"]:.4f}\n')

    # ── Collect validation sample indices ─────────────────────
    all_mics = sorted(os.listdir(os.path.join(args.sim_dir, 'mic')))
    # Take the last n_samples (non-overlapping with training set)
    # Filter out incomplete mic/ref/clean triples
    valid_indices = []
    for f in all_mics:
        idx = f.replace('mic_', '').replace('.wav', '')
        if (os.path.exists(os.path.join(args.sim_dir, 'ref',   f'ref_{idx}.wav')) and
            os.path.exists(os.path.join(args.sim_dir, 'clean', f'clean_{idx}.wav'))):
            valid_indices.append(idx)
    indices = valid_indices[-args.n_samples:]
    print(f'Valid samples: {len(valid_indices)}, evaluating: {len(indices)}\n')

    # ── Method list ───────────────────────────────────────────
    methods = ['noisy']
    if HAS_WEBRTC_NS:
        methods.append('webrtc_ns')
    if HAS_SPEEX_AEC:
        methods.append('webrtc_aec_ns')
    if HAS_RNNOISE:
        methods.append('rnnoise')
    if HAS_RNNOISE and HAS_SPEEX_AEC:
        methods.append('rnnoise_aec')
    methods.append('our_model')

    # Accumulate metrics
    accum = {m: {'si_snr': [], 'erle': []} for m in methods}
    if HAS_PESQ:
        for m in methods: accum[m]['pesq'] = []
    if HAS_STOI:
        for m in methods: accum[m]['stoi'] = []

    # ── Per-sample processing ─────────────────────────────────
    for idx in indices:
        mic_path   = os.path.join(args.sim_dir, 'mic',   f'mic_{idx}.wav')
        ref_path   = os.path.join(args.sim_dir, 'ref',   f'ref_{idx}.wav')
        clean_path = os.path.join(args.sim_dir, 'clean', f'clean_{idx}.wav')

        mic_wav,   _ = sf.read(mic_path,   dtype='float32')
        ref_wav,   _ = sf.read(ref_path,   dtype='float32')
        clean_wav, _ = sf.read(clean_path, dtype='float32')

        # Run all methods
        outputs = {'noisy': mic_wav}
        if HAS_WEBRTC_NS:
            outputs['webrtc_ns']     = webrtc_ns(mic_wav)
        if HAS_SPEEX_AEC:
            outputs['webrtc_aec_ns'] = webrtc_aec_ns(mic_wav, ref_wav)
        if HAS_RNNOISE:
            outputs['rnnoise']       = rnnoise_denoise(mic_wav)
        if HAS_RNNOISE and HAS_SPEEX_AEC:
            outputs['rnnoise_aec']   = rnnoise_aec(mic_wav, ref_wav)
        outputs['our_model'] = our_model_enhance(model, mic_wav, ref_wav, device)

        # Save wavs
        if save_wav:
            wav_dir = os.path.join(args.out_dir, f'sample_{idx}')
            os.makedirs(wav_dir, exist_ok=True)
            sf.write(os.path.join(wav_dir, 'mic_noisy.wav'),      mic_wav,   SR)
            sf.write(os.path.join(wav_dir, 'ref.wav'),            ref_wav,   SR)
            sf.write(os.path.join(wav_dir, 'clean_target.wav'),   clean_wav, SR)
            for name, wav in outputs.items():
                if name == 'noisy': continue
                sf.write(os.path.join(wav_dir, f'{name}.wav'), wav[:len(mic_wav)], SR)

        # Compute metrics
        for method, wav in outputs.items():
            m = calc_metrics(wav, clean_wav, mic_wav)
            for k, v in m.items():
                accum[method][k].append(v)

    # ── Print summary table ───────────────────────────────────
    metric_names = ['si_snr', 'erle']
    if HAS_PESQ:  metric_names.append('pesq')
    if HAS_STOI:  metric_names.append('stoi')

    col_w = 10
    header = f"{'Method':<18}" + ''.join(f"{m.upper():>{col_w}}" for m in metric_names)
    sep    = '-' * len(header)

    print(sep)
    print(header)
    print(sep)

    results_csv = [','.join(['method'] + metric_names)]

    for method in methods:
        means = {k: float(np.nanmean(accum[method][k])) for k in metric_names if k in accum[method]}
        row   = f"{method:<18}" + ''.join(f"{means.get(k, float('nan')):>{col_w}.3f}" for k in metric_names)
        print(row)
        results_csv.append(','.join([method] + [f"{means.get(k, float('nan')):.4f}" for k in metric_names]))

    print(sep)

    # ── Delta vs noisy ────────────────────────────────────────
    print(f"\nImprovement over noisy input (Δ):")
    noisy_means = {k: float(np.nanmean(accum['noisy'][k])) for k in metric_names if k in accum['noisy']}
    for method in methods:
        if method == 'noisy': continue
        means = {k: float(np.nanmean(accum[method][k])) for k in metric_names if k in accum[method]}
        parts = []
        for k in metric_names:
            d = means.get(k, 0) - noisy_means.get(k, 0)
            parts.append(f"{d:+.3f}".rjust(col_w))
        print(f"  {method:<16} Δ" + ''.join(parts))

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(results_csv))
    print(f"\nMetrics saved: {csv_path}")
    if save_wav:
        print(f"Comparison wavs saved: {args.out_dir}/sample_<idx>/")


if __name__ == '__main__':
    main()
