"""
evaluate.py - Model evaluation script

Computes the following metrics:
  - PESQ  (Perceptual Evaluation of Speech Quality, MOS-LQO)
  - STOI  (Short-Time Objective Intelligibility)
  - ERLE  (Echo Return Loss Enhancement, dB)
  - SI-SNR (Scale-Invariant Signal-to-Noise Ratio, dB)

Usage:
  # Evaluate on the validation set
  python3 src/evaluate.py \
      --ckpt checkpoints/best.pth \
      --sim_dir data/simulated \
      --output_dir eval_results/

  # Single-file inference
  python3 src/evaluate.py \
      --ckpt checkpoints/best.pth \
      --mic_file path/to/mic.wav \
      --ref_file path/to/ref.wav \
      --output path/to/enhanced.wav
"""

import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import JointAECNSSystem
from dataset import build_dataloaders, load_wav, normalize, SAMPLE_RATE


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """Compute PESQ MOS (wideband, wb)"""
    try:
        from pesq import pesq
        return pesq(sr, ref, deg, 'wb')
    except Exception as e:
        return float('nan')


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """Compute STOI intelligibility score"""
    try:
        from pystoi import stoi
        return stoi(ref, deg, sr, extended=False)
    except Exception as e:
        return float('nan')


def compute_erle(mic: np.ndarray, enhanced: np.ndarray, eps: float = 1e-8) -> float:
    """
    ERLE = 10 * log10(E[mic^2] / E[enhanced^2])
    Reflects the amount of echo suppression, in dB; higher is better.
    """
    mic_power = np.mean(mic ** 2) + eps
    enh_power = np.mean(enhanced ** 2) + eps
    return 10 * np.log10(mic_power / enh_power)


def compute_si_snr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SNR, in dB"""
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    dot = np.dot(ref, deg)
    proj = dot / (np.dot(ref, ref) + eps) * ref
    noise = deg - proj
    si_snr = 10 * np.log10(
        np.dot(proj, proj) / (np.dot(noise, noise) + eps)
    )
    return float(si_snr)


def load_model(ckpt_path: str, device: str) -> JointAECNSSystem:
    """Load model from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = JointAECNSSystem().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Model loaded from: {ckpt_path}  (epoch={ckpt.get("epoch", "?")})')
    return model


@torch.no_grad()
def enhance_wav(model: JointAECNSSystem,
                mic_wav: np.ndarray,
                ref_wav: np.ndarray,
                device: str,
                chunk_sec: float = 10.0) -> np.ndarray:
    """
    Enhance a time-domain audio signal with support for chunked processing of long audio.

    Args:
        model:     Loaded model
        mic_wav:   [L] float32 numpy microphone signal
        ref_wav:   [L] float32 numpy far-end reference signal
        device:    'cuda' or 'cpu'
        chunk_sec: Chunk duration (seconds) per forward pass, to avoid OOM

    Returns:
        enhanced: [L] float32 numpy
    """
    sr = SAMPLE_RATE
    chunk_len = int(sr * chunk_sec)
    total_len = len(mic_wav)

    # Align lengths
    min_len = min(len(mic_wav), len(ref_wav))
    mic_wav = mic_wav[:min_len]
    ref_wav = ref_wav[:min_len]

    enhanced_chunks = []
    hid_mic, hid_ref = None, None

    for start in range(0, min_len, chunk_len):
        end = min(start + chunk_len, min_len)
        mic_chunk = torch.from_numpy(mic_wav[start:end]).unsqueeze(0).to(device)   # [1, L]
        ref_chunk = torch.from_numpy(ref_wav[start:end]).unsqueeze(0).to(device)   # [1, L]

        # Streaming inference: pass GRU hidden state from the previous chunk
        enh_chunk, (hid_mic, hid_ref) = model(mic_chunk, ref_chunk, hid_mic, hid_ref)
        enhanced_chunks.append(enh_chunk.squeeze(0).cpu().numpy())

    enhanced = np.concatenate(enhanced_chunks, axis=0)
    return enhanced[:total_len]


def evaluate_dataset(model, sim_dir, aec_dir, device, output_dir=None, max_samples=None):
    """Batch-evaluate on a dataset and print summary metrics"""
    _, val_loader = build_dataloaders(
        aec_dir=aec_dir,
        sim_dir=sim_dir,
        batch_size=1,
        num_workers=0,
    )

    results = {
        'pesq': [], 'stoi': [], 'erle': [], 'si_snr': [],
        'pesq_input': [], 'stoi_input': [], 'si_snr_input': []
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    for mic_wav, ref_wav, clean_wav in tqdm(val_loader, desc='Evaluating'):
        # mic_wav: [1, L]
        mic_np   = mic_wav[0].numpy()
        ref_np   = ref_wav[0].numpy()
        clean_np = clean_wav[0].numpy()

        # Enhance
        enhanced_np = enhance_wav(model, mic_np, ref_np, device)

        # Align lengths
        min_len = min(len(clean_np), len(enhanced_np))
        clean_np    = clean_np[:min_len]
        enhanced_np = enhanced_np[:min_len]
        mic_np_cut  = mic_np[:min_len]

        # Normalise amplitude (before evaluation)
        clean_np    = clean_np / (np.abs(clean_np).max() + 1e-8)
        enhanced_np = enhanced_np / (np.abs(enhanced_np).max() + 1e-8)
        mic_np_cut  = mic_np_cut / (np.abs(mic_np_cut).max() + 1e-8)

        # Compute post-enhancement metrics
        results['pesq'].append(compute_pesq(clean_np, enhanced_np))
        results['stoi'].append(compute_stoi(clean_np, enhanced_np))
        results['erle'].append(compute_erle(mic_np_cut, enhanced_np))
        results['si_snr'].append(compute_si_snr(clean_np, enhanced_np))

        # Compute input (unenhanced) metrics as baseline for comparison
        results['pesq_input'].append(compute_pesq(clean_np, mic_np_cut))
        results['stoi_input'].append(compute_stoi(clean_np, mic_np_cut))
        results['si_snr_input'].append(compute_si_snr(clean_np, mic_np_cut))

        # Save enhanced audio samples
        if output_dir and count < 20:
            sf.write(os.path.join(output_dir, f'enhanced_{count:04d}.wav'),
                     enhanced_np, SAMPLE_RATE)
            sf.write(os.path.join(output_dir, f'mic_{count:04d}.wav'),
                     mic_np_cut, SAMPLE_RATE)
            sf.write(os.path.join(output_dir, f'clean_{count:04d}.wav'),
                     clean_np, SAMPLE_RATE)

        count += 1
        if max_samples and count >= max_samples:
            break

    # Aggregate statistics
    def nanmean(lst):
        arr = np.array([x for x in lst if not np.isnan(x)])
        return float(arr.mean()) if len(arr) > 0 else float('nan')

    print('\n' + '='*55)
    print(f"{'Metric':<20} {'Enhanced':>10} {'Input baseline':>10} {'Improvement':>10}")
    print('-'*55)

    pesq_enh   = nanmean(results['pesq'])
    pesq_in    = nanmean(results['pesq_input'])
    stoi_enh   = nanmean(results['stoi'])
    stoi_in    = nanmean(results['stoi_input'])
    sisnr_enh  = nanmean(results['si_snr'])
    sisnr_in   = nanmean(results['si_snr_input'])
    erle_avg   = nanmean(results['erle'])

    print(f'{"PESQ (MOS)":<20} {pesq_enh:>10.3f} {pesq_in:>10.3f} {pesq_enh - pesq_in:>+10.3f}')
    print(f'{"STOI":<20} {stoi_enh:>10.3f} {stoi_in:>10.3f} {stoi_enh - stoi_in:>+10.3f}')
    print(f'{"SI-SNR (dB)":<20} {sisnr_enh:>10.2f} {sisnr_in:>10.2f} {sisnr_enh - sisnr_in:>+10.2f}')
    print(f'{"ERLE (dB)":<20} {erle_avg:>10.2f} {"N/A":>10} {"N/A":>10}')
    print(f'{"Samples":<20} {count:>10}')
    print('='*55)

    return {
        'pesq': pesq_enh, 'stoi': stoi_enh,
        'si_snr': sisnr_enh, 'erle': erle_avg,
        'n_samples': count,
    }


def parse_args():
    p = argparse.ArgumentParser(description='AEC+NS model evaluation')
    p.add_argument('--ckpt',       type=str, required=True, help='Model checkpoint path')
    # Dataset evaluation
    p.add_argument('--aec_dir',    type=str, default=None)
    p.add_argument('--sim_dir',    type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None, help='Directory to save enhanced audio')
    p.add_argument('--max_samples',type=int, default=None, help='Maximum number of samples to evaluate')
    # Single-file inference
    p.add_argument('--mic_file',   type=str, default=None, help='Single-file inference: microphone audio')
    p.add_argument('--ref_file',   type=str, default=None, help='Single-file inference: reference audio')
    p.add_argument('--output',     type=str, default=None, help='Single-file inference: output path')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.ckpt, device)

    # Single-file inference mode
    if args.mic_file and args.ref_file:
        mic_wav = load_wav(args.mic_file).numpy()
        ref_wav = load_wav(args.ref_file).numpy()
        enhanced = enhance_wav(model, mic_wav, ref_wav, device)

        out_path = args.output or 'enhanced.wav'
        sf.write(out_path, enhanced, SAMPLE_RATE)
        print(f'Enhancement complete, output: {out_path}')
        return

    # Dataset evaluation mode
    assert args.aec_dir or args.sim_dir, 'Please provide --aec_dir or --sim_dir'
    evaluate_dataset(
        model,
        sim_dir=args.sim_dir,
        aec_dir=args.aec_dir,
        device=device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
