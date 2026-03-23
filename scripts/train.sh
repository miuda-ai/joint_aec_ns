#!/bin/bash
# train.sh - Training script (nohup friendly, outputs to logs/train.log)

cd "$(dirname "$0")/.."

mkdir -p logs checkpoints

echo "========================================================"
echo "  Training started  $(date)"
echo "  PID: $$"
echo "========================================================"

python3 src/train.py \
    --sim_dir data/simulated \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-3 \
    --conv_channels 64 \
    --gru_hidden 64 \
    --ckpt_dir checkpoints \
    --log_dir logs \
    --save_every 5 \
    --loss_alpha 0.5 --loss_beta 0.3 --loss_gamma 0.2

echo ""
echo "========================================================"
echo "  Training finished  $(date)"
echo "========================================================"
