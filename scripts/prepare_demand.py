"""
prepare_demand.py - Decode DEMAND parquet shards to wav files

Usage:
  python3 scripts/prepare_demand.py \
      --parquet_dir data/demand_parquet \
      --output_dir  data/demand_wav
"""

import os
import io
import argparse
import glob
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from tqdm import tqdm

SR = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet_dir', type=str, default='data/demand_parquet')
    p.add_argument('--output_dir',  type=str, default='data/demand_wav')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, 'data', '*.parquet')))
    assert parquet_files, f'Parquet files not found: {args.parquet_dir}/data/*.parquet'
    print(f'Found {len(parquet_files)} shards')

    total_written = 0
    for pf in parquet_files:
        shard_name = os.path.splitext(os.path.basename(pf))[0]
        table = pq.read_table(pf)
        rows = table.to_pydict()
        file_names = rows['file_name']
        audios = rows['audio']

        for fname, audio in tqdm(zip(file_names, audios), total=len(file_names),
                                  desc=shard_name, leave=False):
            out_path = os.path.join(args.output_dir, f'{shard_name}__{fname}.wav')
            if os.path.exists(out_path):
                total_written += 1
                continue
            try:
                audio_bytes = audio['bytes']
                wav, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
                # Convert to mono if needed
                if wav.ndim == 2:
                    wav = wav.mean(axis=-1)
                # Resample if not 16kHz
                if sr != SR:
                    import torchaudio, torch
                    t = torch.from_numpy(wav).unsqueeze(0)
                    t = torchaudio.functional.resample(t, sr, SR)
                    wav = t.squeeze(0).numpy()
                sf.write(out_path, wav, SR)
                total_written += 1
            except Exception as e:
                print(f'  Skipped {fname}: {e}')

    print(f'\nDone: wrote {total_written} wav files -> {args.output_dir}')


if __name__ == '__main__':
    main()
