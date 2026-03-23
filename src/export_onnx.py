"""
export_onnx.py - Export trained JointAECNSModel to ONNX format

Exports two models:
  1. export/model_full.onnx  - Batch spectrum domain inference (dynamic T dimension)
     Input: mic_stft [1, T, 257, 2], ref_stft [1, T, 257, 2]
     Output: out_stft [1, T, 257, 2]

  2. export/model_stream.onnx - Frame-by-frame streaming inference (T=1, with GRU hidden state)
     Input: mic_stft [1, 1, 257, 2], ref_stft [1, 1, 257, 2],
           hid_mic [1, 1, 64], hid_ref [1, 1, 64]
     Output: out_stft [1, 1, 257, 2], hid_mic_out [1, 1, 64], hid_ref_out [1, 1, 64]

Note: STFT/iSTFT are not included in ONNX (implemented externally with numpy/scipy).

Usage:
  python3 src/export_onnx.py
  python3 src/export_onnx.py --ckpt checkpoints/best.pth --out_dir export
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import JointAECNSModel


# Wrapper classes

class FullModelWrapper(nn.Module):
    """
    Batch/offline inference wrapper (no GRU hidden state input/output, auto-initialized to zero).
    Input: mic_stft [B, T, F, 2], ref_stft [B, T, F, 2]
    Output: out_stft [B, T, F, 2]
    """
    def __init__(self, model: JointAECNSModel):
        super().__init__()
        self.model = model

    def forward(self, mic_stft: torch.Tensor, ref_stft: torch.Tensor) -> torch.Tensor:
        out_stft, _ = self.model(mic_stft, ref_stft, None, None)
        return out_stft


class StreamModelWrapper(nn.Module):
    """
    Streaming frame-by-frame inference wrapper (explicitly passes GRU hidden state).
    Input: mic_stft [B, 1, F, 2], ref_stft [B, 1, F, 2],
          hid_mic [1, B, H], hid_ref [1, B, H]
    Output: out_stft [B, 1, F, 2], hid_mic_out [1, B, H], hid_ref_out [1, B, H]
    """
    def __init__(self, model: JointAECNSModel):
        super().__init__()
        self.model = model

    def forward(self,
                mic_stft: torch.Tensor,
                ref_stft: torch.Tensor,
                hid_mic: torch.Tensor,
                hid_ref: torch.Tensor):
        out_stft, (hid_mic_out, hid_ref_out) = self.model(
            mic_stft, ref_stft, hid_mic, hid_ref
        )
        return out_stft, hid_mic_out, hid_ref_out


# Export functions

def export_full_model(model: JointAECNSModel,
                      out_path: str,
                      freq_bins: int = 257,
                      opset: int = 17) -> None:
    """Export batch inference model (dynamic T)"""
    wrapper = FullModelWrapper(model)
    wrapper.eval()

    # Dummy input (B=1, T=100, F=257, 2)
    T = 100
    mic_dummy = torch.randn(1, T, freq_bins, 2)
    ref_dummy = torch.randn(1, T, freq_bins, 2)

    print(f"  Exporting {out_path} ...")
    # Use dynamo=False to use old path, avoiding torch.export dynamic_axes limitations
    torch.onnx.export(
        wrapper,
        (mic_dummy, ref_dummy),
        out_path,
        opset_version=opset,
        input_names=["mic_stft", "ref_stft"],
        output_names=["out_stft"],
        dynamic_axes={
            "mic_stft": {0: "batch", 1: "time"},
            "ref_stft": {0: "batch", 1: "time"},
            "out_stft": {0: "batch", 1: "time"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported: {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


def export_stream_model(model: JointAECNSModel,
                        out_path: str,
                        freq_bins: int = 257,
                        gru_hidden: int = 64,
                        opset: int = 17) -> None:
    """Export streaming frame-by-frame inference model (T=1, with GRU hidden state)"""
    wrapper = StreamModelWrapper(model)
    wrapper.eval()

    # Dummy input (B=1, T=1, single frame)
    mic_dummy  = torch.randn(1, 1, freq_bins, 2)
    ref_dummy  = torch.randn(1, 1, freq_bins, 2)
    hid_mic_d  = torch.zeros(1, 1, gru_hidden)
    hid_ref_d  = torch.zeros(1, 1, gru_hidden)

    print(f"  Exporting {out_path} ...")
    # Use dynamo=False to use old path
    torch.onnx.export(
        wrapper,
        (mic_dummy, ref_dummy, hid_mic_d, hid_ref_d),
        out_path,
        opset_version=opset,
        input_names=["mic_stft", "ref_stft", "hid_mic", "hid_ref"],
        output_names=["out_stft", "hid_mic_out", "hid_ref_out"],
        dynamic_axes={
            "mic_stft":     {0: "batch"},
            "ref_stft":     {0: "batch"},
            "hid_mic":      {1: "batch"},
            "hid_ref":      {1: "batch"},
            "out_stft":     {0: "batch"},
            "hid_mic_out":  {1: "batch"},
            "hid_ref_out":  {1: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported: {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


# Verification functions

def verify_full_model(model: JointAECNSModel,
                      onnx_path: str,
                      freq_bins: int = 257) -> None:
    """Compare PyTorch and ONNX outputs to verify consistency"""
    print(f"\nVerifying {onnx_path} ...")
    wrapper = FullModelWrapper(model)
    wrapper.eval()

    T = 50
    mic = torch.randn(1, T, freq_bins, 2)
    ref = torch.randn(1, T, freq_bins, 2)

    # PyTorch output
    with torch.no_grad():
        pt_out = wrapper(mic, ref).numpy()

    # ONNX output
    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["out_stft"],
        {"mic_stft": mic.numpy(), "ref_stft": ref.numpy()}
    )[0]

    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    print(f"  PyTorch output shape: {pt_out.shape}")
    print(f"  ONNX   output shape: {ort_out.shape}")
    print(f"  Max error: {max_diff:.2e}  Mean error: {mean_diff:.2e}")
    if max_diff < 1e-4:
        print(f"  Consistency verified (max diff < 1e-4)")
    else:
        print(f"  WARNING: Large error (max diff = {max_diff:.2e}), please check model")


def verify_stream_model(model: JointAECNSModel,
                        onnx_path: str,
                        freq_bins: int = 257,
                        gru_hidden: int = 64,
                        n_frames: int = 20) -> None:
    """Verify streaming model consistency with PyTorch frame-by-frame"""
    print(f"\nVerifying {onnx_path} ...")

    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])

    # Initialize hidden states
    hid_mic_pt = None
    hid_ref_pt = None
    hid_mic_ort = np.zeros((1, 1, gru_hidden), dtype=np.float32)
    hid_ref_ort = np.zeros((1, 1, gru_hidden), dtype=np.float32)

    max_diffs = []
    model.eval()

    for i in range(n_frames):
        mic = torch.randn(1, 1, freq_bins, 2)
        ref = torch.randn(1, 1, freq_bins, 2)

        # PyTorch inference
        with torch.no_grad():
            pt_out, (hid_mic_pt, hid_ref_pt) = model(
                mic, ref, hid_mic_pt, hid_ref_pt
            )

        # ONNX inference
        ort_results = sess.run(
            ["out_stft", "hid_mic_out", "hid_ref_out"],
            {
                "mic_stft": mic.numpy(),
                "ref_stft": ref.numpy(),
                "hid_mic":  hid_mic_ort,
                "hid_ref":  hid_ref_ort,
            }
        )
        ort_out, hid_mic_ort, hid_ref_ort = ort_results

        diff = np.abs(pt_out.numpy() - ort_out).max()
        max_diffs.append(diff)

    overall_max = max(max_diffs)
    print(f"  Frame-by-frame inference {n_frames} frames, max error: {overall_max:.2e}")
    if overall_max < 1e-4:
        print(f"  Streaming consistency verified (all frames max diff < 1e-4)")
    else:
        print(f"  WARNING: Large streaming error (max diff = {overall_max:.2e}), please check model")


# Main function

def parse_args():
    p = argparse.ArgumentParser(description="Export JointAECNSModel to ONNX (streaming mode)")
    p.add_argument("--ckpt",       type=str, default="checkpoints/best.pth",
                   help="checkpoint path")
    p.add_argument("--hparams",    type=str, default="checkpoints/hparams.json",
                   help="hparams.json path")
    p.add_argument("--out_dir",    type=str, default="export",
                   help="ONNX output directory")
    p.add_argument("--opset",      type=int, default=17,
                   help="ONNX opset version (recommended: 17)")
    p.add_argument("--conv_channels", type=int, default=None,
                   help="override conv_channels")
    p.add_argument("--gru_hidden",   type=int, default=None,
                   help="override gru_hidden")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Read hyperparameters
    if os.path.exists(args.hparams):
        with open(args.hparams) as f:
            hp = json.load(f)
        freq_bins    = hp.get("freq_bins", 257)
        conv_channels = hp.get("conv_channels", 64)
        gru_hidden   = hp.get("gru_hidden", 64)
        n_conv_layers = hp.get("n_conv_layers", 3)
    else:
        freq_bins, conv_channels, gru_hidden, n_conv_layers = 257, 64, 64, 3

    # Override with command line arguments if provided
    if args.conv_channels is not None:
        conv_channels = args.conv_channels
    if args.gru_hidden is not None:
        gru_hidden = args.gru_hidden

    print(f"Hyperparameters: freq_bins={freq_bins}, conv_channels={conv_channels}, "
          f"gru_hidden={gru_hidden}, n_conv_layers={n_conv_layers}")

    # Load model
    device = "cpu"   # ONNX export must be on CPU
    model = JointAECNSModel(
        freq_bins=freq_bins,
        conv_channels=conv_channels,
        gru_hidden=gru_hidden,
        n_conv_layers=n_conv_layers,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # checkpoint may be wrapped under "model_state" key
    state = ckpt.get("model_state", ckpt)
    # Remove "model." prefix (if checkpoint came from JointAECNSSystem)
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v
    # Filter out stft_proc related buffers (not part of JointAECNSModel)
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in new_state.items() if k in model_keys}
    model.load_state_dict(filtered_state)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    n_params = model.count_parameters()
    print(f"Model loaded: epoch={epoch}, val_loss={val_loss:.4f}, params={n_params:,}\n")

    # Export streaming model only (model_stream.onnx)
    print("Exporting streaming model (model_stream.onnx)")
    stream_path = os.path.join(args.out_dir, "model_stream.onnx")
    export_stream_model(model, stream_path,
                       freq_bins=freq_bins, gru_hidden=gru_hidden,
                       opset=args.opset)

    # Verify
    print("\n" + "="*50)
    print("Verifying ONNX output consistency with PyTorch")
    print("="*50)
    verify_stream_model(model, stream_path,
                       freq_bins=freq_bins, gru_hidden=gru_hidden)

    # Print I/O specs
    print("\n" + "="*50)
    print("ONNX model spec summary")
    print("="*50)
    sess = ort.InferenceSession(stream_path, providers=["CPUExecutionProvider"])
    print(f"\n[model_stream]")
    print("  Inputs:")
    for inp in sess.get_inputs():
        print(f"    {inp.name}: {inp.shape} ({inp.type})")
    print("  Outputs:")
    for out in sess.get_outputs():
        print(f"    {out.name}: {out.shape} ({out.type})")
    print(f"  File size: {os.path.getsize(stream_path)/1024:.1f} KB")

    print("\nONNX export complete!")
    print(f"  {stream_path}")


if __name__ == "__main__":
    main()
