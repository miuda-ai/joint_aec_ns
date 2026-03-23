"""
generate_sim_data.py - Generate simulated AEC training data using pyroomacoustics

How it works:
  1. Randomly sample room parameters (dimensions, RT60)
  2. Generate a room impulse response (RIR)
  3. Convolve far-end reference signal with RIR → echo signal
  4. mic = clean_speech + echo + noise (DEMAND real noise or Gaussian white noise)
  5. Save mic / ref / clean wav triples

Requires clean speech material:
  - Use LibriSpeech or any custom speech file collection
  - Accepts any directory of mono wav files as speech source

Usage:
  # White noise only (legacy behaviour)
  python3 src/generate_sim_data.py \
      --speech_dir data/librispeech_wav \
      --output_dir data/simulated \
      --n_samples 30000 \
      --n_jobs 16

  # With DEMAND real noise (recommended)
  python3 src/generate_sim_data.py \
      --speech_dir data/librispeech_wav \
      --noise_dir data/demand \
      --output_dir data/simulated \
      --n_samples 30000 \
      --n_jobs 16
"""

import os
import sys
import argparse
import random
import glob
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

SAMPLE_RATE = 16000
CLIP_LEN = 4 * SAMPLE_RATE   # 4 seconds


def load_wav_np(path: str, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Load wav as numpy float32, resampling if necessary."""
    try:
        data, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=-1)
        if orig_sr != sr:
            import torchaudio, torch
            tensor = torch.from_numpy(data).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, orig_sr, sr)
            data = tensor.squeeze(0).numpy()
        return data
    except Exception:
        return None


def random_crop_or_pad(wav: np.ndarray, length: int) -> np.ndarray:
    if len(wav) >= length:
        start = random.randint(0, len(wav) - length)
        return wav[start: start + length]
    else:
        return np.pad(wav, (0, length - len(wav)))


def generate_rir(room_dims, source_pos, mic_pos, rt60: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a room impulse response using an impulse source."""
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(
        room_dims,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        air_absorption=True,
        ray_tracing=False,
    )
    # Unit impulse source for RIR computation
    impulse = np.zeros(128, dtype=np.float32)
    impulse[0] = 1.0
    room.add_source(source_pos, signal=impulse)
    mic_array = np.array(mic_pos).reshape(3, 1)
    room.add_microphone(mic_array)
    room.simulate()
    rir = room.rir[0][0]   # single mic, single source
    return rir.astype(np.float32)


def load_noise_wav(noise_files, clip_len, sr):
    """Load a random noise clip from the DEMAND file list; returns None on failure."""
    if not noise_files:
        return None
    path = random.choice(noise_files)
    wav = load_wav_np(path, sr)
    if wav is None or len(wav) < sr:
        return None
    return random_crop_or_pad(wav, clip_len)


def simulate_one(args_tuple):
    """Generate a single simulated sample (suitable for multiprocessing)."""
    idx, speech_files, noise_files, output_dir, sr, clip_len = args_tuple

    # Skip if already exists (supports resumable generation)
    stem = f'{idx:06d}'
    if (os.path.exists(os.path.join(output_dir, 'mic',   f'mic_{stem}.wav')) and
        os.path.exists(os.path.join(output_dir, 'ref',   f'ref_{stem}.wav')) and
        os.path.exists(os.path.join(output_dir, 'clean', f'clean_{stem}.wav'))):
        return True

    try:
        # Randomly select near-end speech (target speaker)
        clean_path = random.choice(speech_files)
        clean_wav = load_wav_np(clean_path, sr)
        if clean_wav is None or len(clean_wav) < sr:
            return False

        # Randomly select far-end speech (the other side of the call)
        ref_path = random.choice(speech_files)
        ref_wav = load_wav_np(ref_path, sr)
        if ref_wav is None or len(ref_wav) < sr:
            return False

        clean_wav = random_crop_or_pad(clean_wav, clip_len)
        ref_wav   = random_crop_or_pad(ref_wav,   clip_len)

        # Random room parameters
        lx = random.uniform(3.0, 8.0)
        ly = random.uniform(3.0, 8.0)
        lz = random.uniform(2.5, 4.0)
        rt60 = random.uniform(0.15, 0.6)   # reverberation time 150–600 ms

        # Loudspeaker position (source of echo)
        spk_x = random.uniform(0.3, lx - 0.3)
        spk_y = random.uniform(0.3, ly - 0.3)
        spk_z = random.uniform(0.5, lz - 0.5)

        # Microphone position
        mic_x = random.uniform(0.3, lx - 0.3)
        mic_y = random.uniform(0.3, ly - 0.3)
        mic_z = random.uniform(0.8, 1.5)

        room_dims  = [lx, ly, lz]
        source_pos = [spk_x, spk_y, spk_z]
        mic_pos    = [mic_x, mic_y, mic_z]

        # Generate RIR
        rir = generate_rir(room_dims, source_pos, mic_pos, rt60, sr)

        # Echo = ref convolved with RIR
        echo = np.convolve(ref_wav, rir)[:clip_len]

        # Random Signal-to-Echo Ratio (SER): 0–20 dB
        # clean >= echo, preventing the model from learning "attenuate everything"
        ser_db = random.uniform(0, 20)
        ser = 10 ** (ser_db / 20)

        clean_rms = np.sqrt(np.mean(clean_wav ** 2) + 1e-8)
        echo_rms  = np.sqrt(np.mean(echo ** 2) + 1e-8)
        echo_scaled = echo * (clean_rms / (echo_rms * ser + 1e-8))

        # Background noise: when noise_files provided, 70% DEMAND real noise / 30% white noise;
        # without noise_files, always white noise
        snr_db = random.uniform(10, 30)
        snr = 10 ** (snr_db / 20)

        demand_noise = None
        if noise_files and random.random() < 0.7:
            demand_noise = load_noise_wav(noise_files, clip_len, sr)

        if demand_noise is not None:
            noise = demand_noise
        else:
            noise = np.random.randn(clip_len).astype(np.float32)

        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
        noise_scaled = noise * (clean_rms / (noise_rms * snr + 1e-8))

        # Synthesise microphone signal
        mic_wav = clean_wav + echo_scaled + noise_scaled

        # Peak normalisation to prevent clipping
        peak = max(np.abs(mic_wav).max(), np.abs(ref_wav).max(), np.abs(clean_wav).max()) + 1e-8
        if peak > 0.99:
            mic_wav   = mic_wav   / peak * 0.95
            ref_wav   = ref_wav   / peak * 0.95
            clean_wav = clean_wav / peak * 0.95

        # Save
        stem = f'{idx:06d}'
        sf.write(os.path.join(output_dir, 'mic',   f'mic_{stem}.wav'),   mic_wav,   sr)
        sf.write(os.path.join(output_dir, 'ref',   f'ref_{stem}.wav'),   ref_wav,   sr)
        sf.write(os.path.join(output_dir, 'clean', f'clean_{stem}.wav'), clean_wav, sr)

        return True

    except Exception as e:
        return False


def parse_args():
    p = argparse.ArgumentParser(description='Generate simulated AEC training data')
    p.add_argument('--speech_dir', type=str, required=True,
                   help='Clean speech directory (contains .wav files, searched recursively)')
    p.add_argument('--noise_dir',  type=str, default=None,
                   help='DEMAND real noise directory (contains .wav files); omit to use white noise only')
    p.add_argument('--output_dir', type=str, default='data/simulated',
                   help='Output directory')
    p.add_argument('--n_samples',  type=int, default=15000,
                   help='Number of samples to generate')
    p.add_argument('--n_jobs',     type=int, default=4,
                   help='Number of parallel worker processes')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    for sub in ['mic', 'ref', 'clean']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Collect speech files
    speech_files = sorted(glob.glob(os.path.join(args.speech_dir, '**', '*.wav'), recursive=True))
    if not speech_files:
        speech_files = sorted(glob.glob(os.path.join(args.speech_dir, '*.wav')))
    assert speech_files, f'No .wav files found in {args.speech_dir}'

    # Collect noise files (optional)
    noise_files = []
    if args.noise_dir:
        noise_files = sorted(glob.glob(os.path.join(args.noise_dir, '**', '*.wav'), recursive=True))
        if not noise_files:
            noise_files = sorted(glob.glob(os.path.join(args.noise_dir, '*.wav')))

    print(f'Speech files found:  {len(speech_files)}')
    print(f'Noise files found:   {len(noise_files)} {"(DEMAND)" if noise_files else "(will use white noise)"}')
    print(f'Samples to generate: {args.n_samples}')
    print(f'Output directory:    {args.output_dir}')
    print(f'Worker processes:    {args.n_jobs}')

    # Build task list
    tasks = [
        (i, speech_files, noise_files, args.output_dir, SAMPLE_RATE, CLIP_LEN)
        for i in range(args.n_samples)
    ]

    success = 0
    fail = 0

    if args.n_jobs <= 1:
        for task in tqdm(tasks, desc='Generating'):
            if simulate_one(task):
                success += 1
            else:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(simulate_one, t): t[0] for t in tasks}
            with tqdm(total=len(tasks), desc='Generating') as pbar:
                for future in as_completed(futures):
                    if future.result():
                        success += 1
                    else:
                        fail += 1
                    pbar.update(1)

    print(f'\nDone: {success} succeeded / {fail} failed')
    print(f'Data saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
