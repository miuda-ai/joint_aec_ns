#!/bin/bash
# retrain_demand.sh - Retrain with DEMAND noise data
# Usage: nohup bash scripts/retrain_demand.sh > logs/retrain_demand.log 2>&1 &

set -e
cd /home/rzl/workspace/rs/kaka

echo "========================================="
echo " Step 1: Decode DEMAND parquet → wav"
echo "========================================="
python3 scripts/prepare_demand.py \
    --parquet_dir data/demand_parquet \
    --output_dir  data/demand_wav

DEMAND_COUNT=$(ls data/demand_wav/*.wav 2>/dev/null | wc -l)
echo "DEMAND wav files: $DEMAND_COUNT"

echo ""
echo "========================================="
echo " Step 2: Regenerate 30K simulated data (with DEMAND noise)"
echo "========================================="
rm -rf data/simulated/mic data/simulated/ref data/simulated/clean
python3 src/generate_sim_data.py \
    --speech_dir data/librispeech_wav \
    --noise_dir  data/demand_wav \
    --output_dir data/simulated \
    --n_samples  30000 \
    --n_jobs     16 \
    --seed       123

echo ""
echo "========================================="
echo " Step 3: Retrain (lr=3e-4, patience=10)"
echo "========================================="
PYTORCH_ALLOC_CONF=expandable_segments:True python3 src/train.py \
    --sim_dir      data/simulated \
    --epochs       100 \
    --batch_size   32 \
    --num_workers  4 \
    --lr           3e-4 \
    --patience     10 \
    --conv_channels 64 \
    --gru_hidden   64 \
    --clip_sec     2.0 \
    --ckpt_dir     checkpoints \
    --log_dir      logs \
    --save_every   5 \
    --loss_alpha 0.5 --loss_beta 0.3 --loss_gamma 0.2

echo "Done!"
