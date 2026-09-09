#!/usr/bin/env bash
set -euo pipefail

py="${PYTHON:-python}"
case "${1:-}" in
  check)
    "$py" - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
    ;;
  smoke)
    shift
    "$py" studies/cnn_patch_energy_control/evaluate_gpu.py --smoke --output outputs/patch_energy_control_smoke "$@"
    ;;
  main)
    shift
    "$py" studies/cnn_patch_energy_control/evaluate_gpu.py --output outputs/patch_energy_control "$@"
    ;;
  report)
    shift
    "$py" studies/cnn_patch_energy_control/summarize.py "${1:-outputs/patch_energy_control}"
    ;;
  *)
    echo 'Usage: bash run_patch_energy_control.sh [check|smoke|main|report] [options]'
    exit 2
    ;;
esac
