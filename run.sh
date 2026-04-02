#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$SCRIPT_DIR/venv/bin/python"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/venv." >&2
    echo "Run 'bash setup.sh' first." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: ./run.sh ROOT [options]" >&2
    echo "Example: ./run.sh sample_media --log-level INFO" >&2
    echo "Example: ./run.sh --version" >&2
    exit 1
fi

exec "$PYTHON_EXE" "$MAIN_SCRIPT" "$@"
