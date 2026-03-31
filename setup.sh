#!/usr/bin/env bash
# setup.sh — create virtual environment and install all dependencies.
#
# Usage:
#   bash setup.sh
#
# After setup, activate the environment with:
#   source venv/bin/activate

set -euo pipefail

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"

echo "=========================================="
echo "  CEPOL Transcripter — Environment Setup  "
echo "=========================================="

# ── 1. Check Python version ─────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.10+ and retry." >&2
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version : $PY_VERSION"
if [[ "${PY_VERSION%%.*}" -lt 3 || ( "${PY_VERSION%%.*}" -eq 3 && "${PY_VERSION##*.}" -lt 10 ) ]]; then
    echo "ERROR: Python 3.10 or newer is required." >&2
    exit 1
fi

# ── 2. Check ffmpeg ──────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "WARNING: ffmpeg is not installed or not in PATH."
    echo "Install it with:"
    echo "   Ubuntu/Debian : sudo apt install ffmpeg"
    echo "   macOS (brew)  : brew install ffmpeg"
    echo ""
fi

# ── 3. Create virtual environment ───────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at ./$VENV_DIR — skipping creation."
else
    echo "Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── 4. Upgrade pip ──────────────────────────────────────────────────────────
echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

# ── 5. Install dependencies ──────────────────────────────────────────────────
echo "Installing requirements ..."
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Run the transcriber:"
echo "    python main.py /path/to/media/root"
echo "=========================================="
