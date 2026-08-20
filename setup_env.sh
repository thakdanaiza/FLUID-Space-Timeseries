#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

if [[ "${1:-}" == "--recreate" ]]; then
  rm -rf -- ".venv"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
else
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.12 from:"
  echo "https://www.python.org/downloads/macos/"
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
  echo "FLUID-Space requires Python 3.11, 3.12, or 3.13."
  echo "Python 3.12 from Python.org is recommended."
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import tkinter' >/dev/null 2>&1; then
  echo "Tkinter is unavailable in this Python installation."
  echo "Install Python 3.12 using the official Python.org macOS installer."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[1/4] Creating .venv..."
  "$PYTHON_BIN" -m venv ".venv"
else
  echo "[1/4] Using existing .venv..."
fi

echo "[2/4] Updating pip..."
".venv/bin/python" -m pip install --upgrade pip

echo "[3/4] Installing FLUID-Space packages..."
".venv/bin/python" -m pip install -r requirements.txt

echo "[4/4] Checking the application..."
".venv/bin/python" app.py --check

echo
echo "Environment check passed."
echo "Start the application with: bash start_ui.sh"
