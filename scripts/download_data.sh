#!/bin/bash
# download_data.sh - Data download script
#
# Usage:
#   bash scripts/download_data.sh
#
# Proxy (if needed for GitHub access):
#   export PROXY=socks5://192.168.3.152:1080
#   bash scripts/download_data.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
PROXY="${PROXY:-socks5://192.168.3.152:1080}"

echo "========================================================"
echo "  AEC+NS Model - Data Download Script"
echo "  Data directory: $DATA_DIR"
echo "  Proxy: $PROXY"
echo "========================================================"

# ── Tool check ─────────────────────────────────────────────
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] Tool not found: $1"
        exit 1
    fi
}

# ── 1. AEC-Challenge dataset (git-lfs, requires proxy) ────
download_aec_challenge() {
    echo ""
    echo "─────────────────────────────────────────────────────"
    echo "  [1/2] AEC-Challenge Synthetic Dataset"
    echo "─────────────────────────────────────────────────────"

    AEC_DIR="$DATA_DIR/aec_challenge"

    if [ -d "$AEC_DIR/synthetic/nearend_mic_signal" ] && \
       [ "$(ls -A "$AEC_DIR/synthetic/nearend_mic_signal" 2>/dev/null | wc -l)" -gt 100 ]; then
        echo "  [SKIP] AEC-Challenge already exists"
        return
    fi

    # Check git-lfs
    if ! git lfs version &>/dev/null; then
        echo "  Installing git-lfs..."
        sudo apt-get install -y git-lfs -q
        git lfs install
    fi

    mkdir -p "$AEC_DIR"
    echo "  Cloning AEC-Challenge (with git-lfs, ~10GB, proxy required)..."
    echo "  This may take a while depending on network speed..."

    # Set proxy for git
    GIT_PROXY_CMD="git -c http.proxy=$PROXY -c https.proxy=$PROXY"

    # Shallow clone first to test connectivity
    if ! $GIT_PROXY_CMD clone \
        --depth 1 \
        --no-checkout \
        "https://github.com/microsoft/AEC-Challenge" \
        "$AEC_DIR/repo_tmp" 2>/dev/null; then
        echo ""
        echo "  [ERROR] Failed to clone AEC-Challenge"
        echo ""
        echo "  Manual download:"
        echo "  1. Install git-lfs: sudo apt install git-lfs && git lfs install"
        echo "  2. Configure proxy: git config --global http.proxy $PROXY"
        echo "  3. Clone: git clone https://github.com/microsoft/AEC-Challenge $AEC_DIR/repo"
        echo "  4. Copy data to: $AEC_DIR/synthetic/"
        echo "     Expected structure:"
        echo "       $AEC_DIR/synthetic/nearend_mic_signal/*.wav"
        echo "       $AEC_DIR/synthetic/farend_speech/*.wav"
        echo "       $AEC_DIR/synthetic/nearend_speech/*.wav"
        return 1
    fi

    rm -rf "$AEC_DIR/repo_tmp"

    # Full clone with lfs
    GIT_LFS_SKIP_SMUDGE=0 $GIT_PROXY_CMD clone \
        "https://github.com/microsoft/AEC-Challenge" \
        "$AEC_DIR/repo"

    # Organize directory
    if [ -d "$AEC_DIR/repo/datasets/synthetic" ]; then
        cp -r "$AEC_DIR/repo/datasets/synthetic/"* "$AEC_DIR/synthetic/"
        echo "  [DONE] AEC-Challenge downloaded"
        echo "  Samples: $(ls "$AEC_DIR/synthetic/nearend_mic_signal/" | wc -l)"
    fi
}

# ── 2. LibriSpeech (speech source for simulated data) ──────
download_librispeech() {
    echo ""
    echo "─────────────────────────────────────────────────────"
    echo "  [2/2] LibriSpeech (for generating simulated data)"
    echo "─────────────────────────────────────────────────────"

    LIBRI_DIR="$DATA_DIR/librispeech"

    if [ -d "$LIBRI_DIR" ] && \
       [ "$(find "$LIBRI_DIR" -name "*.flac" | wc -l)" -gt 1000 ]; then
        echo "  [SKIP] LibriSpeech already exists ($(find "$LIBRI_DIR" -name "*.flac" | wc -l) files)"
        return
    fi

    mkdir -p "$LIBRI_DIR"

    # LibriSpeech dev-clean (~337MB)
    LIBRI_URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"

    echo "  Downloading LibriSpeech dev-clean (~337MB)..."
    if curl --proxy "$PROXY" -L -C - \
        -o "$LIBRI_DIR/dev-clean.tar.gz" \
        "$LIBRI_URL" 2>&1; then
        echo "  Extracting..."
        tar -xzf "$LIBRI_DIR/dev-clean.tar.gz" -C "$LIBRI_DIR" --strip-components=1
        rm "$LIBRI_DIR/dev-clean.tar.gz"
        echo "  [DONE] LibriSpeech dev-clean downloaded"
        echo "  Files: $(find "$LIBRI_DIR" -name "*.flac" | wc -l)"
    else
        echo ""
        echo "  [WARNING] LibriSpeech download failed (proxy may be required)"
        echo ""
        echo "  Manual download options:"
        echo "  Option A - LibriSpeech dev-clean:"
        echo "    wget -c $LIBRI_URL -O $LIBRI_DIR/dev-clean.tar.gz"
        echo "    tar -xzf $LIBRI_DIR/dev-clean.tar.gz -C $LIBRI_DIR"
        echo ""
        echo "  Option B - Use local speech files:"
        echo "    Put .wav or .flac files in any directory,"
        echo "    then use --speech_dir to point to that directory"
    fi
}

# ── 3. Convert flac → wav (pyroomacoustics needs wav) ───────
convert_flac_to_wav() {
    LIBRI_DIR="$DATA_DIR/librispeech"
    WAV_DIR="$DATA_DIR/librispeech_wav"

    FLAC_COUNT=$(find "$LIBRI_DIR" -name "*.flac" 2>/dev/null | wc -l)
    if [ "$FLAC_COUNT" -eq 0 ]; then
        return
    fi

    if [ -d "$WAV_DIR" ] && [ "$(find "$WAV_DIR" -name "*.wav" | wc -l)" -gt 1000 ]; then
        echo "  [SKIP] wav conversion already done"
        return
    fi

    echo ""
    echo "  Converting flac to wav ($FLAC_COUNT files)..."
    mkdir -p "$WAV_DIR"

    python3 - <<'PYEOF'
import os, glob, soundfile as sf, numpy as np
from tqdm import tqdm

libri_dir = os.environ.get('LIBRI_DIR', 'data/librispeech')
wav_dir   = os.environ.get('WAV_DIR',   'data/librispeech_wav')
os.makedirs(wav_dir, exist_ok=True)

flac_files = glob.glob(os.path.join(libri_dir, '**', '*.flac'), recursive=True)
print(f'Converting {len(flac_files)} flac files...')
for i, f in enumerate(tqdm(flac_files)):
    try:
        data, sr = sf.read(f, dtype='float32')
        if data.ndim == 2: data = data.mean(-1)
        out_path = os.path.join(wav_dir, f'{i:06d}.wav')
        sf.write(out_path, data, sr)
    except Exception:
        pass
print(f'Done: {wav_dir}')
PYEOF

    export LIBRI_DIR="$LIBRI_DIR"
    export WAV_DIR="$WAV_DIR"
    echo "  [DONE] flac → wav conversion complete"
}

# ── 4. Generate simulated data ────────────────────────────────
generate_simulated_data() {
    echo ""
    echo "─────────────────────────────────────────────────────"
    echo "  Generate Simulated AEC Data (pyroomacoustics)"
    echo "─────────────────────────────────────────────────────"

    SIM_DIR="$DATA_DIR/simulated"
    SIM_COUNT=$(find "$SIM_DIR/mic" -name "*.wav" 2>/dev/null | wc -l)

    if [ "$SIM_COUNT" -ge 5000 ]; then
        echo "  [SKIP] Simulated data already exists ($SIM_COUNT samples)"
        return
    fi

    # Find speech source directory
    SPEECH_DIR=""
    for candidate in \
        "$DATA_DIR/librispeech_wav" \
        "$DATA_DIR/librispeech" \
        "$DATA_DIR/voicebank/clean_trainset_28spk_wav"; do
        if [ -d "$candidate" ] && \
           [ "$(find "$candidate" -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l)" -gt 100 ]; then
            SPEECH_DIR="$candidate"
            break
        fi
    done

    if [ -z "$SPEECH_DIR" ]; then
        echo "  [WARNING] Speech source directory not found"
        echo "  Please manually specify speech source and run:"
        echo "    python3 src/generate_sim_data.py \\"
        echo "        --speech_dir /path/to/your/clean_wavs \\"
        echo "        --output_dir $SIM_DIR \\"
        echo "        --n_samples 15000 \\"
        echo "        --n_jobs 8"
        return
    fi

    echo "  Speech source: $SPEECH_DIR"
    echo "  Target: 15000 samples"
    N_JOBS=$(nproc)
    echo "  Parallel jobs: $N_JOBS"

    python3 "$PROJECT_DIR/src/generate_sim_data.py" \
        --speech_dir "$SPEECH_DIR" \
        --output_dir "$SIM_DIR" \
        --n_samples 15000 \
        --n_jobs "$N_JOBS"
}

# ── Main ────────────────────────────────────────────────────
echo ""
echo "Starting data preparation..."

download_aec_challenge
download_librispeech
convert_flac_to_wav
generate_simulated_data

echo ""
echo "========================================================"
echo "  Data preparation complete!"
echo ""
echo "  Data directory structure:"
echo "  $DATA_DIR/"
if [ -d "$DATA_DIR/aec_challenge/synthetic" ]; then
    MIC_COUNT=$(ls "$DATA_DIR/aec_challenge/synthetic/nearend_mic_signal/" 2>/dev/null | wc -l)
    echo "  ├── aec_challenge/synthetic/  ($MIC_COUNT samples)"
fi
if [ -d "$DATA_DIR/simulated/mic" ]; then
    SIM_COUNT=$(ls "$DATA_DIR/simulated/mic/" 2>/dev/null | wc -l)
    echo "  └── simulated/               ($SIM_COUNT samples)"
fi
echo ""
echo "  Start training:"
echo "  python3 src/train.py \\"
echo "      --sim_dir $DATA_DIR/simulated \\"
echo "      --epochs 100"
echo "========================================================"
