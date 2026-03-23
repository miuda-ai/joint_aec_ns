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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import JointAECNSModel, STFTProcessor

SR = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320
FREQ_BINS = 257
GRU_HIDDEN = 64


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
    stft_proc = STFTProcessor(N_FFT, HOP_LENGTH, WIN_LENGTH, SR)

    if use_onnx:
        print(f"Using ONNX model: {args.onnx}, gru_hidden={gru_hidden}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            args.onnx,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )
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

    # Generate demo for 3 samples
    sim_dir = "data/simulated"

    for i in range(3):
        mic_path = f"{sim_dir}/mic/mic_{i:06d}.wav"
        ref_path = f"{sim_dir}/ref/ref_{i:06d}.wav"
        clean_path = f"{sim_dir}/clean/clean_{i:06d}.wav"

        mic_wav, _ = sf.read(mic_path)
        ref_wav, _ = sf.read(ref_path)
        clean_wav, _ = sf.read(clean_path)

        L = min(len(mic_wav), len(ref_wav), len(clean_wav))
        mic_wav, ref_wav, clean_wav = mic_wav[:L], ref_wav[:L], clean_wav[:L]

        # Inference
        if use_onnx:
            enhanced = onnx_stream_inference(sess, stft_proc, mic_wav, ref_wav, gru_hidden=gru_hidden)
        else:
            enhanced = pytorch_stream_inference(model, stft_proc, mic_wav, ref_wav)

        sf.write(f"{out_dir}/mic_{i:02d}.wav", mic_wav, SR)
        sf.write(f"{out_dir}/clean_{i:02d}.wav", clean_wav, SR)
        sf.write(f"{out_dir}/enhanced_{i:02d}.wav", enhanced, SR)
        print(f"Sample {i}: saved mic, clean, enhanced")

    print(f"\nDemo files saved to {out_dir}/")
    print("  mic_00.wav, mic_01.wav, mic_02.wav     - Noisy microphone input")
    print("  clean_00.wav, clean_01.wav, clean_02.wav - Clean reference")
    print("  enhanced_00.wav, enhanced_01.wav, enhanced_02.wav - Enhanced output")


if __name__ == "__main__":
    main()
