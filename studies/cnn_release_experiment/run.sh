#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
mode="${1:-main}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  smoke) .venv/bin/python experiment.py --output outputs/smoke --epochs 2 --blocks 1 --tasks two_concepts --architectures TinyCNN --conditions template_release --release-start 0 --release-end 2 --checkpoints 1 2 "$@" ;;
  pilot) .venv/bin/python experiment.py --output outputs/pilot --blocks 1 "$@" ;;
  main) .venv/bin/python experiment.py "$@" ;;
  report) .venv/bin/python summarize.py "$@" ;;
  *) echo 'Usage: ./run.sh [smoke|pilot|main|report] [options]'; exit 2 ;;
esac
