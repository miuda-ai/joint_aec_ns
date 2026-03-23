"""
train.py - Main training script

Usage:
  python3 src/train.py \
      --aec_dir data/aec_challenge/synthetic \
      --sim_dir data/simulated \
      --epochs 100 \
      --batch_size 16 \
      --lr 1e-3

Resume from checkpoint:
  python3 src/train.py --resume checkpoints/best.pth ...
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# allow running from the project root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import JointAECNSModel, STFTProcessor
from dataset import build_dataloaders
from losses import JointLoss


def parse_args():
    p = argparse.ArgumentParser(description='AEC+NS joint model training')
    # data
    p.add_argument('--aec_dir', type=str, default=None,
                   help='AEC-Challenge synthetic data directory (containing nearend_mic_signal/ and other subdirectories)')
    p.add_argument('--sim_dir', type=str, default=None,
                   help='Simulated data directory (containing mic/ ref/ clean/ subdirectories)')
    # model
    p.add_argument('--freq_bins',     type=int, default=257)
    p.add_argument('--conv_channels', type=int, default=32)
    p.add_argument('--gru_hidden',    type=int, default=32)
    p.add_argument('--n_conv_layers', type=int, default=3)
    p.add_argument('--n_fft',         type=int, default=512)
    p.add_argument('--hop_length',    type=int, default=160,  help='10ms @ 16kHz')
    p.add_argument('--win_length',    type=int, default=320,  help='20ms @ 16kHz')
    # training
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--clip_sec',   type=float, default=4.0,  help='duration of each training sample (seconds)')
    # loss weights
    p.add_argument('--loss_alpha', type=float, default=0.5,  help='SI-SNR weight')
    p.add_argument('--loss_beta',  type=float, default=0.3,  help='frequency-domain MSE weight')
    p.add_argument('--loss_gamma', type=float, default=0.2,  help='magnitude L1 weight')
    # save / resume
    p.add_argument('--ckpt_dir',  type=str, default='checkpoints')
    p.add_argument('--log_dir',   type=str, default='logs')
    p.add_argument('--resume',    type=str, default=None, help='checkpoint path to resume from')
    p.add_argument('--save_every',type=int, default=5,    help='save checkpoint every N epochs')
    # misc
    p.add_argument('--patience', type=int, default=5, help='ReduceLROnPlateau patience')
    p.add_argument('--seed',  type=int, default=42)
    p.add_argument('--amp',   action='store_true', default=False, help='mixed precision training (BF16); FP32 is faster and free of inf grad, disabled by default')
    # Note: streaming mode is now the default (no --batch option)
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, stft_proc,
                    device, scaler, epoch, amp_dtype, use_amp):
    model.train()
    total_loss = 0.0
    loss_items = {}
    n_batches = len(loader)
    n_valid = 0  # number of steps actually updated (after NaN skips)

    for i, (mic_wav, ref_wav, clean_wav) in enumerate(loader):
        mic_wav   = mic_wav.to(device)
        ref_wav   = ref_wav.to(device)
        clean_wav = clean_wav.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # forward pass
            enhanced_wav, _ = model(mic_wav, ref_wav)

            # compute target STFT (for frequency-domain loss)
            target_stft = stft_proc.stft(clean_wav)    # [B, T, F, 2]
            pred_stft   = stft_proc.stft(enhanced_wav) # [B, T, F, 2]

            loss, loss_dict = criterion(enhanced_wav, clean_wav, pred_stft, target_stft)

        # NaN/Inf guard (forward): skip anomalous batches to avoid corrupting parameters and Adam momentum
        if not torch.isfinite(loss):
            print(f'  [Epoch {epoch}] step {i}: loss={loss.item()}, skipping', flush=True)
            del loss, enhanced_wav, pred_stft, target_stft
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # gradient NaN/Inf guard
            grads_ok = all(torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [Epoch {epoch}] step {i}: inf/nan grad, skipping', flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # gradient NaN/Inf guard: skip parameter update on inf grad (avoid corrupting Adam momentum)
            grads_ok = all(torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [Epoch {epoch}] step {i}: inf/nan grad, skipping', flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_valid += 1
        for k, v in loss_dict.items():
            loss_items[k] = loss_items.get(k, 0.0) + v

        if (i + 1) % 50 == 0:
            avg = total_loss / n_valid if n_valid > 0 else float('nan')
            print(f'  [Epoch {epoch}] {i+1}/{n_batches}  avg_loss={avg:.4f}', flush=True)

    n = max(n_valid, 1)
    avg_loss = total_loss / n
    avg_items = {k: v / n for k, v in loss_items.items()}
    return avg_loss, avg_items


def train_one_epoch_streaming(model, loader, optimizer, criterion, stft_proc,
                              device, scaler, epoch, amp_dtype, use_amp):
    """Streaming mode: frame-by-frame training."""
    model.train()
    total_loss = 0.0
    loss_items = {}
    n_batches = len(loader)
    n_valid = 0

    for i, (mic_wav, ref_wav, clean_wav) in enumerate(loader):
        mic_wav   = mic_wav.to(device)
        ref_wav   = ref_wav.to(device)
        clean_wav = clean_wav.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # STFT first
            mic_stft = stft_proc.stft(mic_wav)    # [B, T, F, 2]
            ref_stft = stft_proc.stft(ref_wav)
            clean_stft = stft_proc.stft(clean_wav)

            B, T, F, _ = mic_stft.shape
            # Initialize hidden states
            hid_mic = torch.zeros(1, B, model.gru_hidden, device=device)
            hid_ref = torch.zeros(1, B, model.gru_hidden, device=device)

            # Frame-by-frame forward
            outputs = []
            for t in range(T):
                mic_f = mic_stft[:, t:t+1]   # [B, 1, F, 2]
                ref_f = ref_stft[:, t:t+1]
                out_f, (hid_mic, hid_ref) = model(mic_f, ref_f, hid_mic, hid_ref)
                outputs.append(out_f)

            # Concatenate all frames
            out_stft = torch.cat(outputs, dim=1)  # [B, T, F, 2]

            # iSTFT to time domain
            enhanced_wav = stft_proc.istft(out_stft, length=mic_wav.shape[1])

            # Compute loss in time domain
            target_stft = clean_stft
            pred_stft = stft_proc.stft(enhanced_wav)
            loss, loss_dict = criterion(enhanced_wav, clean_wav, pred_stft, target_stft)

        # NaN/Inf guard
        if not torch.isfinite(loss):
            print(f'  [Epoch {epoch}] step {i}: loss={loss.item()}, skipping', flush=True)
            del loss, enhanced_wav, pred_stft, target_stft
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads_ok = all(torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [Epoch {epoch}] step {i}: inf/nan grad, skipping', flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grads_ok = all(torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [Epoch {epoch}] step {i}: inf/nan grad, skipping', flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_valid += 1
        for k, v in loss_dict.items():
            loss_items[k] = loss_items.get(k, 0.0) + v

        if (i + 1) % 50 == 0:
            avg = total_loss / n_valid if n_valid > 0 else float('nan')
            print(f'  [Epoch {epoch}] {i+1}/{n_batches}  avg_loss={avg:.4f}', flush=True)

    n = max(n_valid, 1)
    avg_loss = total_loss / n
    avg_items = {k: v / n for k, v in loss_items.items()}
    return avg_loss, avg_items


@torch.no_grad()
def validate_streaming(model, loader, criterion, stft_proc, device):
    """Validation for streaming mode."""
    model.eval()
    total_loss = 0.0
    loss_items = {}

    for mic_wav, ref_wav, clean_wav in loader:
        mic_wav   = mic_wav.to(device)
        ref_wav   = ref_wav.to(device)
        clean_wav = clean_wav.to(device)

        # STFT
        mic_stft = stft_proc.stft(mic_wav)
        ref_stft = stft_proc.stft(ref_wav)
        clean_stft = stft_proc.stft(clean_wav)

        B, T, F, _ = mic_stft.shape
        hid_mic = torch.zeros(1, B, model.gru_hidden, device=device)
        hid_ref = torch.zeros(1, B, model.gru_hidden, device=device)

        outputs = []
        for t in range(T):
            mic_f = mic_stft[:, t:t+1]
            ref_f = ref_stft[:, t:t+1]
            out_f, (hid_mic, hid_ref) = model(mic_f, ref_f, hid_mic, hid_ref)
            outputs.append(out_f)

        out_stft = torch.cat(outputs, dim=1)
        enhanced_wav = stft_proc.istft(out_stft, length=mic_wav.shape[1])

        target_stft = clean_stft
        pred_stft = stft_proc.stft(enhanced_wav)
        loss, loss_dict = criterion(enhanced_wav, clean_wav, pred_stft, target_stft)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_items[k] = loss_items.get(k, 0.0) + v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_items.items()}


@torch.no_grad()
def validate(model, loader, criterion, stft_proc, device):
    model.eval()
    total_loss = 0.0
    loss_items = {}

    for mic_wav, ref_wav, clean_wav in loader:
        mic_wav   = mic_wav.to(device)
        ref_wav   = ref_wav.to(device)
        clean_wav = clean_wav.to(device)

        enhanced_wav, _ = model(mic_wav, ref_wav)
        target_stft = stft_proc.stft(clean_wav)
        pred_stft   = stft_proc.stft(enhanced_wav)

        loss, loss_dict = criterion(enhanced_wav, clean_wav, pred_stft, target_stft)
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_items[k] = loss_items.get(k, 0.0) + v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_items.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    torch.save({
        'epoch':        epoch,
        'val_loss':     val_loss,
        'model_state':  model.state_dict(),
        'optim_state':  optimizer.state_dict(),
        'sched_state':  scheduler.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])
    return ckpt['epoch'], ckpt['val_loss']


def main():
    args = parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── directories ───────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # ── data ──────────────────────────────────────────────────
    clip_len = int(16000 * args.clip_sec)
    train_loader, val_loader = build_dataloaders(
        aec_dir=args.aec_dir,
        sim_dir=args.sim_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_len=clip_len,
    )

    # ── model ─────────────────────────────────────────────────
    # Always use streaming mode (JointAECNSModel)
    print("Using streaming mode (JointAECNSModel)")
    model = JointAECNSModel(
        freq_bins=args.freq_bins,
        conv_channels=args.conv_channels,
        gru_hidden=args.gru_hidden,
        n_conv_layers=args.n_conv_layers,
    ).to(device)
    # Create standalone STFT processor and move to device
    stft_proc = STFTProcessor(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sample_rate=16000,
    ).to(device)

    print(f'Model parameter count: {model.count_parameters():,}')

    # ── loss / optimizer / scheduler ──────────────────────────
    criterion = JointLoss(args.loss_alpha, args.loss_beta, args.loss_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup for 3 epochs (lr linearly rises from 1/5 to target lr)
    # to avoid large Adam cold-start steps causing oscillation.
    # SequentialLR runs LinearLR for warmup_epochs steps, then hands off to ReduceLROnPlateau.
    warmup_epochs = 3
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, end_factor=1.0,
                                 total_iters=warmup_epochs)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                          patience=args.patience, min_lr=1e-6)
    # scheduler kept for checkpoint save/resume compatibility (only plateau_scheduler is saved)
    scheduler = plateau_scheduler

    # ── AMP setup ─────────────────────────────────────────────
    # BF16: dynamic range 3.4e38, far larger than FP16's 65504, eliminating overflow → NaN
    # RTX 2080 Ti supports BF16; BF16 does not require GradScaler
    use_amp = args.amp and device == 'cuda'
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    scaler = torch.amp.GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None
    print(f'AMP dtype: {amp_dtype}  GradScaler: {scaler is not None}')

    # ── resume from checkpoint ────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f'Resumed from {args.resume}, epoch={start_epoch}, best_val={best_val_loss:.4f}')

    # ── TensorBoard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=args.log_dir)

    # save hyperparameters
    with open(os.path.join(args.ckpt_dir, 'hparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── training loop ─────────────────────────────────────────
    print(f'\nStart training, total {args.epochs} epochs\n')

    # Always use streaming mode
    train_fn = train_one_epoch_streaming
    val_fn = validate_streaming

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # train
        train_loss, train_items = train_fn(
            model, train_loader, optimizer, criterion, stft_proc,
            device, scaler, epoch, amp_dtype, use_amp)

        # validate
        val_loss, val_items = val_fn(
            model, val_loader, criterion, stft_proc, device)

        # scheduler: use LinearLR during warmup, then ReduceLROnPlateau
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
              f't={elapsed:.1f}s')

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        for k, v in train_items.items():
            writer.add_scalar(f'Train/{k}', v, epoch)
        for k, v in val_items.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.ckpt_dir, 'best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f'  ★ Saved best checkpoint  val_loss={val_loss:.4f}')

        # periodic save
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path)

    writer.close()
    print(f'\nTraining complete! best val_loss={best_val_loss:.4f}')
    print(f'Best checkpoint: {os.path.join(args.ckpt_dir, "best.pth")}')


if __name__ == '__main__':
    main()
