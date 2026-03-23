"""
benchmark.py - Single-threaded CPU latency comparison

Measures processing latency per 10ms frame (160 samples @ 16kHz) for each method.
Methods:
  1. WebRTC NS                  - webrtc_noise_gain.AudioProcessor.Process10ms()
  2. WebRTC AEC+NS              - speexdsp.EchoCanceller.process() + WebRTC NS
  3. RNNoise                    - pyrnnoise.RNNoise.denoise_chunk() (480 samples/30ms)
  4. RNNoise + AEC              - SpeexDSP AEC + RNNoise
  5. Our model (ONNX FP32)      - ONNX Runtime CPU streaming frame-by-frame (single-threaded)

Measurement:
  - warmup: 10 frames
  - timing: 200 frames
  - report: mean +/- std ms/frame, RTF (Real-Time Factor)

Usage:
  python3 src/benchmark.py
  python3 src/benchmark.py --warmup 20 --n_frames 500
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import onnxruntime as ort

# Optional dependencies
try:
    from webrtc_noise_gain import AudioProcessor as WebRTCAP
    HAS_WEBRTC_NS = True
except ImportError:
    HAS_WEBRTC_NS = False
    print("[WARNING] webrtc_noise_gain not installed. pip install webrtc-noise-gain")

try:
    from speexdsp import EchoCanceller
    HAS_SPEEX_AEC = True
except ImportError:
    HAS_SPEEX_AEC = False
    print("[WARNING] speexdsp not installed. pip install speexdsp")

try:
    from pyrnnoise import RNNoise
    HAS_RNNOISE = True
except ImportError:
    HAS_RNNOISE = False
    print("[WARNING] pyrnnoise not installed. pip install pyrnnoise")

SR = 16000
FRAME = 160        # 10ms @ 16kHz
RNN_FRAME = 480    # RNNoise fixed frame length 30ms, 3 x 10ms sub-frames per frame


# Measurement utility

def measure(fn, n_warmup: int = 10, n_measure: int = 200) -> dict:
    """Run fn() multiple times and return latency statistics (ms)"""
    # Warmup
    for _ in range(n_warmup):
        fn()

    # Timing
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times)
    frame_duration_ms = FRAME / SR * 1000.0   # 10.0 ms
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "p50":    float(np.percentile(arr, 50)),
        "p95":    float(np.percentile(arr, 95)),
        "max":    float(arr.max()),
        "rtf":    float(arr.mean() / frame_duration_ms),
    }


# Benchmark implementations for each method

def bench_webrtc_ns(warmup: int, n_frames: int) -> dict:
    """WebRTC NS pure noise suppression"""
    if not HAS_WEBRTC_NS:
        return None
    ap = WebRTCAP(SR, 0)
    # Generate one real PCM frame each time (random noise as placeholder)
    chunk_i16 = (np.random.randn(FRAME).clip(-1, 1) * 32767).astype(np.int16).tobytes()

    def fn():
        ap.Process10ms(chunk_i16)

    return measure(fn, warmup, n_frames)


def bench_webrtc_aec_ns(warmup: int, n_frames: int) -> dict:
    """SpeexDSP AEC + WebRTC NS in series, processing one 10ms frame at a time"""
    if not HAS_SPEEX_AEC or not HAS_WEBRTC_NS:
        return None
    ec = EchoCanceller.create(FRAME, 2048, SR)
    ap = WebRTCAP(SR, 0)
    mic_chunk = (np.random.randn(FRAME).clip(-1, 1) * 32767).astype(np.int16).tobytes()
    ref_chunk = (np.random.randn(FRAME).clip(-1, 1) * 32767).astype(np.int16).tobytes()

    def fn():
        aec_out = ec.process(mic_chunk, ref_chunk)
        ap.Process10ms(aec_out)

    return measure(fn, warmup, n_frames)


def bench_rnnoise(warmup: int, n_frames: int) -> dict:
    """RNNoise pure noise suppression (30ms frame, equivalent per-10ms latency divided by 3)"""
    if not HAS_RNNOISE:
        return None
    rnn = RNNoise(SR)
    # RNNoise fixed 480 samples / 30ms, equivalent to processing 3 x 10ms frames
    chunk_f32 = np.random.randn(RNN_FRAME).astype(np.float32)

    def fn():
        # denoise_chunk takes 30ms block, we record 30ms latency and convert to 10ms
        list(rnn.denoise_chunk(chunk_f32, partial=False))

    result = measure(fn, warmup, n_frames)
    # Convert: 30ms block -> 10ms equivalent
    frame_duration_ms = FRAME / SR * 1000.0
    result["mean_per10ms"] = result["mean"] / 3.0
    result["std_per10ms"]  = result["std"]  / 3.0
    result["rtf"] = result["mean_per10ms"] / frame_duration_ms
    # For unified table, replace mean/std with per-10ms numbers
    result["mean_raw"] = result["mean"]
    result["mean"] = result["mean_per10ms"]
    result["std"]  = result["std_per10ms"]
    return result


def bench_rnnoise_aec(warmup: int, n_frames: int) -> dict:
    """SpeexDSP AEC (10ms) + RNNoise (30ms), using 10ms as base unit"""
    if not HAS_SPEEX_AEC or not HAS_RNNOISE:
        return None

    ec = EchoCanceller.create(FRAME, 2048, SR)
    rnn = RNNoise(SR)

    mic_chunk = (np.random.randn(FRAME).clip(-1, 1) * 32767).astype(np.int16).tobytes()
    ref_chunk = (np.random.randn(FRAME).clip(-1, 1) * 32767).astype(np.int16).tobytes()
    rnn_buf = np.zeros(RNN_FRAME, dtype=np.float32)
    buf_pos = [0]

    def fn():
        # AEC: per 10ms
        aec_out = ec.process(mic_chunk, ref_chunk)
        aec_f32 = np.frombuffer(aec_out, dtype=np.int16).astype(np.float32) / 32767.0
        # Fill RNNoise buffer
        rnn_buf[buf_pos[0]: buf_pos[0] + FRAME] = aec_f32
        buf_pos[0] += FRAME
        if buf_pos[0] >= RNN_FRAME:
            list(rnn.denoise_chunk(rnn_buf.copy(), partial=False))
            buf_pos[0] = 0

    # Pre-fill 2 frames for warmup buffer
    for _ in range(3):
        fn()
    buf_pos[0] = 0

    # Reset RNNoise
    rnn = RNNoise(SR)
    buf_pos[0] = 0

    return measure(fn, warmup, n_frames)


def bench_onnx_model(onnx_path: str,
                     gru_hidden: int,
                     freq_bins: int,
                     warmup: int,
                     n_frames: int) -> dict:
    """ONNX Runtime CPU streaming frame-by-frame inference (single-threaded)"""
    if not os.path.exists(onnx_path):
        print(f"  [WARNING] {onnx_path} does not exist, skipping")
        return None

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )

    mic_f = np.random.randn(1, 1, freq_bins, 2).astype(np.float32)
    ref_f = np.random.randn(1, 1, freq_bins, 2).astype(np.float32)
    hid_mic = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    hid_ref = np.zeros((1, 1, gru_hidden), dtype=np.float32)

    def fn():
        nonlocal hid_mic, hid_ref
        results = sess.run(
            ["out_stft", "hid_mic_out", "hid_ref_out"],
            {"mic_stft": mic_f, "ref_stft": ref_f,
             "hid_mic": hid_mic, "hid_ref": hid_ref}
        )
        _, hid_mic, hid_ref = results

    return measure(fn, warmup, n_frames)


# Main function

def parse_args():
    p = argparse.ArgumentParser(description="Single-threaded CPU latency comparison benchmark")
    p.add_argument("--onnx",       type=str, default="export/model_stream.onnx")
    p.add_argument("--hparams",    type=str, default="checkpoints/hparams.json")
    p.add_argument("--warmup",     type=int, default=10, help="Number of warmup frames")
    p.add_argument("--n_frames",   type=int, default=200, help="Number of timing frames")
    return p.parse_args()


def fmt_result(r: dict) -> str:
    if r is None:
        return "N/A"
    return f"{r['mean']:.3f} +/- {r['std']:.3f}"


def main():
    args = parse_args()

    # Read hyperparameters
    if os.path.exists(args.hparams):
        with open(args.hparams) as f:
            hp = json.load(f)
        freq_bins  = hp.get("freq_bins", 257)
        gru_hidden = hp.get("gru_hidden", 64)
    else:
        freq_bins, gru_hidden = 257, 64

    frame_duration_ms = FRAME / SR * 1000.0   # 10.0 ms

    print("=" * 65)
    print("Single-threaded CPU latency comparison benchmark")
    print(f"Frame length: {FRAME} samples ({frame_duration_ms:.1f} ms @ {SR} Hz)")
    print(f"Warmup: {args.warmup} frames  Timing: {args.n_frames} frames")
    print("=" * 65)

    results = {}

    # 1. WebRTC NS
    print("\n[1/5] WebRTC NS ...", flush=True)
    r = bench_webrtc_ns(args.warmup, args.n_frames)
    results["WebRTC NS"] = r
    if r:
        print(f"      {r['mean']:.3f} +/- {r['std']:.3f} ms/frame  RTF={r['rtf']:.4f}")

    # 2. WebRTC AEC+NS
    print("[2/5] WebRTC AEC+NS ...", flush=True)
    r = bench_webrtc_aec_ns(args.warmup, args.n_frames)
    results["WebRTC AEC+NS"] = r
    if r:
        print(f"      {r['mean']:.3f} +/- {r['std']:.3f} ms/frame  RTF={r['rtf']:.4f}")

    # 3. RNNoise
    print("[3/5] RNNoise ...", flush=True)
    r = bench_rnnoise(args.warmup, args.n_frames)
    results["RNNoise"] = r
    if r:
        print(f"      {r['mean']:.3f} +/- {r['std']:.3f} ms/frame (equivalent 10ms)  RTF={r['rtf']:.4f}")

    # 4. RNNoise + AEC
    print("[4/5] RNNoise + AEC ...", flush=True)
    r = bench_rnnoise_aec(args.warmup, args.n_frames)
    results["RNNoise + AEC"] = r
    if r:
        print(f"      {r['mean']:.3f} +/- {r['std']:.3f} ms/frame  RTF={r['rtf']:.4f}")

    # 5. Our model ONNX FP32
    print("[5/5] Our model (ONNX FP32) ...", flush=True)
    r = bench_onnx_model(args.onnx, gru_hidden, freq_bins,
                          args.warmup, args.n_frames)
    results["Our model (FP32)"] = r
    if r:
        print(f"      {r['mean']:.3f} +/- {r['std']:.3f} ms/frame  RTF={r['rtf']:.4f}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Method':<22}  {'mean+/-std (ms/frame)':>22}  {'RTF':>8}  {'p95 (ms)':>10}")
    print("-" * 65)

    method_order = [
        "WebRTC NS",
        "WebRTC AEC+NS",
        "RNNoise",
        "RNNoise + AEC",
        "Our model (FP32)",
    ]

    for name in method_order:
        r = results.get(name)
        if r is None:
            print(f"  {name:<20}  {'N/A':>22}  {'N/A':>8}  {'N/A':>10}")
        else:
            ms_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
            print(f"  {name:<20}  {ms_str:>22}  {r['rtf']:>8.4f}  {r['p95']:>9.3f}ms")

    print("=" * 65)
    print(f"  RTF < 1.0 means real-time feasible  Frame duration = {frame_duration_ms:.1f} ms")
    print()

    # Output Markdown table (for README)
    print("Markdown table (copy to README):")
    print()
    print("| Method | Latency (ms/frame) | RTF | p95 (ms) |")
    print("|------|----------------|-----|---------|")
    rnnoise_note = "dagger"
    for name in method_order:
        r = results.get(name)
        note = rnnoise_note if "RNNoise" in name else ""
        if r is None:
            print(f"| {name}{note} | N/A | N/A | N/A |")
        else:
            ms_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
            print(f"| {name}{note} | {ms_str} | {r['rtf']:.4f} | {r['p95']:.3f} |")
    print()
    print("> Single-threaded CPU, frame size 10ms (160 samples @ 16kHz), warmup=10, n=200.")
    print("> dagger RNNoise frame size 30ms, converted to equivalent 10ms latency.")
    print()

    # Return result dict for README usage
    return results


if __name__ == "__main__":
    main()
