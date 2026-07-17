#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────
# Whisper Patois Fine-Tune — Environment Setup
# ──────────────────────────────────────────────────

VENV_DIR="${VENV_DIR:-venv}"

echo "=== Creating Python virtual environment: $VENV_DIR ==="
python3 -m venv "$VENV_DIR"

echo "=== Activating virtual environment ==="
source "$VENV_DIR/bin/activate"

# Determine CUDA version (if any) for PyTorch install
echo "=== Checking CUDA availability ==="
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    echo "  Detected CUDA $CUDA_VERSION"
    case "$CUDA_VERSION" in
        11.8)
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
            ;;
        12.1|12.2|12.3|12.4|12.5|12.6)
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
            ;;
        *)
            echo "  CUDA $CUDA_VERSION not in pinned list; trying default index."
            PYTORCH_INDEX=""
            ;;
    esac
else
    echo "  No CUDA detected. Installing CPU-only PyTorch."
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

if [ -n "${PYTORCH_INDEX:-}" ]; then
    echo "=== Installing PyTorch with CUDA support ==="
    pip install --upgrade pip
    pip install torch --index-url "$PYTORCH_INDEX"
else
    echo "=== Installing PyTorch (default index) ==="
    pip install --upgrade pip
    pip install torch
fi

echo "=== Installing remaining dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Activate the environment with:  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Place audio files in data/raw/"
echo "  2. python scripts/prepare_data.py --raw_dir data/raw --model_size medium"
echo "  3. Correct transcripts in data/transcripts/"
echo "  4. python scripts/prepare_data.py --skip_pseudo"
echo "  5. python scripts/train.py --config config.yaml"
