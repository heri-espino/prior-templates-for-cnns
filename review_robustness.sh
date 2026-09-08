#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${PYTHON:-}" ]]; then py="$PYTHON"
elif [[ -x cnn_release_experiment/.venv/bin/python ]]; then py=cnn_release_experiment/.venv/bin/python
elif [[ -x studies/cnn_release_experiment/.venv/bin/python ]]; then py=studies/cnn_release_experiment/.venv/bin/python
else py=python3
fi
exec "$py" analysis/patch_robustness_gpu_001/review.py
