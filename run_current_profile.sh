#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "FLUID-Space is not installed yet. Starting setup..."
  bash setup_env.sh
fi

exec ".venv/bin/python" run_profile.py --profile current_baseline
