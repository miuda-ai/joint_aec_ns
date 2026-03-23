# JointAEC-NS: Lightweight Causal Joint AEC+NS Model

> 0.108M parameters, single-threaded CPU RTF=0.015, val_loss=-7.149. Supports embedded real-time streaming deployment.

---

## Model Overview

JointAEC-NS is a **lightweight causal** joint acoustic echo cancellation and noise suppression model

- **Fully causal**: No future-frame dependency, 10ms frame delay, naturally supports real-time streaming inference
- **Parameters**: ~0.108M (FP32 ~440 KB ONNX)
- **Architecture**: Causal Conv2D frequency-domain feature extraction → Unidirectional GRU temporal modeling → Lightweight attention ref interaction → Complex Mask
- **Joint design**: Single set of weights for both AEC + NS, no need to cascade two independent modules
- **Training data**: Synthetic simulated data (reverberation + echo + noise), 48GB

### Architecture Diagram

```
mic_wav [B,L]  ref_wav [B,L]
      │               │
    STFT            STFT
      │               │
 [B,T,257,2]   [B,T,257,2]
      └───────┬────────
         [B,4,T,257]
              │
      CausalConv2D × 3
              │
       FreqMeanPool
              │
     GRU_mic   GRU_ref
              │
      LightAttention
              │
          Fusion
              │
         MaskHead
              │
       [B,T,257,2] mask
              │
       mic_stft × mask
              │
            iSTFT
              │
      enhanced_wav [B,L]
```

---

## Performance Metrics (vs baselines, validation set 50 samples)

| Method | SI-SNR (↑) | ERLE (↑) | PESQ (↑) | STOI (↑) |
|--------|-------------|---------|---------|---------|
| Noisy input | 8.499 | 0.000 | 1.464 | 0.845 |
| **Our model** | **14.308** | **1.174** | **1.865** | **0.897** |

> Model: epoch=95, val_loss=-7.149, parameters 108,466.
> ONNX: export/model_stream.onnx (~440 KB FP32).
> Data: Last 50 samples of synthetic simulated validation set (non-overlapping with training set).
> Metrics: SI-SNR (dB, higher is better), ERLE (dB), PESQ-WB (MOS, higher is better), STOI (0~1, higher is better).
> Note: Our model is the only method that improves ALL metrics over noisy input (SI-SNR +5.8 dB, PESQ +0.4, STOI +0.05). Joint design enables single-model AEC+NS with streaming support.

---

## Latency Comparison (Single-threaded CPU)

| Method | Latency (ms/frame) | RTF | p95 (ms) |
|------|----------------|-----|---------|
| WebRTC NS | 0.024 ± 0.027 | 0.0024 | 0.069 |
| WebRTC AEC+NS | 0.041 ± 0.026 | 0.0041 | 0.077 |
| RNNoise dagger | 0.148 ± 0.008 | 0.0148 | 0.509 |
| RNNoise + AEC dagger | 0.172 ± 0.208 | 0.0172 | 0.470 |
| **Our model (FP32)** | **0.152 ± 0.002** | **0.0152** | **0.156** |

> Single-threaded CPU, frame size 10ms (160 samples @ 16 kHz), warmup=10, n=200.
> RTF = processing time / audio duration, <1.0 means real-time feasible (this model RTF ≈ 0.015, far below realtime threshold).
> dagger RNNoise frame size 30ms, converted to equivalent 10ms latency.
> Test machine: CPU-only, single-threaded, ONNX Runtime 1.23.2.
> Our model: (conv_channels=48, gru_hidden=48), ONNX export at export/model_stream.onnx.

---

## Quick Start

### Installation

```bash
git clone <repo>
cd kaka
pip install torch torchaudio numpy scipy soundfile onnxruntime
pip install pesq pystoi webrtc-noise-gain speexdsp pyrnnoise  # for evaluation/baseline comparison
```

### Training

```bash
# 1. Generate synthetic simulated data (requires LibriSpeech + DEMAND)
python3 src/generate_sim_data.py

# 2. Train model
python3 src/train.py \
    --sim_dir data/simulated \
    --ckpt_dir checkpoints \
    --log_dir  logs \
    --conv_channels 48 --gru_hidden 48 \
    --epochs 100 --batch_size 48 --lr 1e-3 \
    --loss_alpha 0.5 --loss_beta 0.3 --loss_gamma 0.2

# TensorBoard: tensorboard --logdir logs
```

### Performance Evaluation (vs baselines)

```bash
python3 src/compare_webrtc.py \
    --ckpt checkpoints/best.pth \
    --n_samples 50 \
    --out_dir results
```

### ONNX Export

```bash
# Export to ONNX (streaming inference)
python3 src/export_onnx.py \
    --ckpt    checkpoints/best.pth \
    --hparams checkpoints/hparams.json \
    --out_dir export \
    --conv_channels 48 --gru_hidden 48

# Output:
#   export/model_stream.onnx  Streaming frame-by-frame inference (~440 KB)
```

### ONNX Inference Test

```bash
# Streaming inference test
python3 src/test_onnx.py \
    --onnx  export/model_stream.onnx \
    --audio results/sample_029997/mic_noisy.wav \
    --ref   results/sample_029997/ref.wav \
    --out   export/onnx_output.wav
```

### Benchmark

```bash
python3 src/benchmark.py \
    --onnx  export/model_stream.onnx \
    --hparams checkpoints/hparams.json \
    --warmup 10 --n_frames 200
```

### ONNX Streaming Inference (Embedded Deployment)

```python
import numpy as np
import onnxruntime as ort

# Load ONNX
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess = ort.InferenceSession("export/model_stream.onnx",
                            sess_options=sess_opts,
                            providers=["CPUExecutionProvider"])

# Initialize GRU hidden states
hid_mic = np.zeros((1, 1, 48), dtype=np.float32)
hid_ref = np.zeros((1, 1, 48), dtype=np.float32)

# Per-frame processing (10ms = 160 samples)
def process_frame(mic_stft_frame, ref_stft_frame):
    global hid_mic, hid_ref
    out_stft, hid_mic, hid_ref = sess.run(
        ["out_stft", "hid_mic_out", "hid_ref_out"],
        {
            "mic_stft": mic_stft_frame,   # [1, 1, 257, 2]
            "ref_stft": ref_stft_frame,   # [1, 1, 257, 2]
            "hid_mic":  hid_mic,
            "hid_ref":  hid_ref,
        }
    )
    return out_stft  # [1, 1, 257, 2]
```

## Model Hyperparameters

| Parameter | Value | Description |
|------|-----|------|
| freq_bins | 257 | STFT frequency bins (N_FFT=512) |
| conv_channels | 48 | Conv2D channel count |
| gru_hidden | 48 | GRU hidden dimension |
| n_conv_layers | 3 | Number of causal Conv2D layers |
| n_fft | 512 | FFT size |
| hop_length | 160 | Frame shift (10ms @ 16kHz) |
| win_length | 320 | Window length (20ms, Hann window) |
| Parameters | 108,466 | ~0.108M |
| FP32 size | ~440 KB | ONNX file size |

---

## License
- MIT License
- Miuda.ai

## Authors
- <shenjindi@miuda.ai>
