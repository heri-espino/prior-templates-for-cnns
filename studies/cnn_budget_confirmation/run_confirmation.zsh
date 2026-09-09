#!/usr/bin/env zsh
# Run the frozen prospective intervention-budget confirmation end to end.
# Uses a user-space Conda environment; no administrator privileges are needed.
#
# Usage:
#   zsh studies/cnn_budget_confirmation/run_confirmation.zsh
#
# Optional environment variables:
#   CNN_ENV_NAME=prior-templates-cnns
#   OUTPUT_ROOT=results/budget_confirmation_001
#   DEVICE=cuda
#   BATCH_SIZE=64
#   THREADS=2
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

set -e
set -u
setopt pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Relative by default so Windows conda.exe also works when launched from WSL.
OUTPUT_ROOT="${OUTPUT_ROOT:-results/budget_confirmation_001}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THREADS="${THREADS:-2}"
CNN_ENV_NAME="${CNN_ENV_NAME:-prior-templates-cnns}"

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
  print -u2 "DEVICE must be 'cuda' or 'cpu'; got: $DEVICE"
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  print -u2 "git is required to record the frozen protocol provenance."
  exit 2
fi

# Find/create the user's Conda environment and expose cnn_python. This avoids
# shell activation and does not require administrator privileges.
source "$REPO_ROOT/scripts/use_conda_env.zsh"

TRAIN_ROOT="$OUTPUT_ROOT/training"
PATCH_ROOT="$OUTPUT_ROOT/patch_eval"
ENERGY_ROOT="$OUTPUT_ROOT/energy_eval"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis/budget_confirmation"
MANIFEST="$OUTPUT_ROOT/execution_manifest.json"
PROTOCOL="studies/cnn_budget_confirmation/PROTOCOL.md"

mkdir -p "$OUTPUT_ROOT"
REPO_HEAD="$(git rev-parse HEAD)"
PROTOCOL_COMMIT="$(git log -n 1 --format=%H -- "$PROTOCOL")"
PROTOCOL_GIT_BLOB="$(git hash-object "$PROTOCOL")"

cnn_python - "$MANIFEST" "$REPO_HEAD" "$PROTOCOL_COMMIT" "$PROTOCOL_GIT_BLOB" "$DEVICE" "$BATCH_SIZE" "$THREADS" <<'PY'
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import scipy
import torch

manifest=Path(sys.argv[1])
repo_head,protocol_commit,protocol_git_blob,device,batch_size,threads=sys.argv[2:]
protocol=Path('studies/cnn_budget_confirmation/PROTOCOL.md')
files=[
    protocol,
    Path('studies/cnn_release_experiment/core.py'),
    Path('studies/cnn_release_experiment/experiment.py'),
    Path('studies/cnn_patch_robustness/evaluate_gpu.py'),
    Path('studies/cnn_patch_energy_control/evaluate_gpu.py'),
    Path('studies/cnn_budget_confirmation/analyze.py'),
    Path('studies/cnn_budget_confirmation/run_confirmation.zsh'),
    Path('scripts/use_conda_env.zsh'),
]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record={
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'repo_head_at_launch': repo_head,
    'protocol_commit': protocol_commit,
    'protocol_git_blob': protocol_git_blob,
    'protocol_sha256': sha(protocol),
    'device': device,
    'batch_size': int(batch_size),
    'training_threads': int(threads),
    'python': sys.version,
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'source_sha256': {str(p): sha(p) for p in files},
    'frozen_design': {
        'task': 'two_concepts',
        'architectures': ['TinyCNN','TwoLayerCNN'],
        'conditions': ['template_retention_1','template_release'],
        'blocks': list(range(4000,4020)),
        'epochs': 200,
        'release_start': 10,
        'release_end': 80,
        'checkpoint_epochs': [1,5,10,20,40,80,120,160,200],
        'intervention_budgets': [1,2,4,8],
        'rankings': ['contrast','auroc','validation_patch'],
        'primary': 'TinyCNN / contrast / selected-fidelity budget contrast B',
    },
}
if manifest.exists():
    old=json.loads(manifest.read_text())
    immutable=['protocol_commit','protocol_sha256','device','batch_size','training_threads','source_sha256','frozen_design']
    changed=[k for k in immutable if old.get(k)!=record.get(k)]
    if changed:
        raise SystemExit('Existing execution manifest is incompatible ('+', '.join(changed)+'). Use a new OUTPUT_ROOT.')
else:
    tmp=manifest.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n')
    tmp.replace(manifest)
print('Execution manifest:',manifest)
print('Protocol commit:',protocol_commit)
print('Conda Python:',sys.executable)
print('Protocol SHA-256:',record['protocol_sha256'])
PY

print ""
print "[1/4] Training 80 fresh models on blocks 4000-4019..."
cnn_python studies/cnn_release_experiment/experiment.py \
  --output "$TRAIN_ROOT" \
  --epochs 200 \
  --blocks 20 \
  --start-block 4000 \
  --tasks two_concepts \
  --architectures TinyCNN TwoLayerCNN \
  --conditions template_retention_1 template_release \
  --release-start 10 \
  --release-end 80 \
  --checkpoints 1 5 10 20 40 80 120 160 200 \
  --threads "$THREADS"

print ""
print "[2/4] Evaluating selected fidelity, random-control U, rankings, and k={1,2,4,8}..."
cnn_python studies/cnn_patch_robustness/evaluate_gpu.py \
  --input "$TRAIN_ROOT" \
  --output "$PATCH_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --epochs 200

print ""
print "[3/4] Evaluating validation-energy-matched controls..."
cnn_python studies/cnn_patch_energy_control/evaluate_gpu.py \
  --input "$TRAIN_ROOT" \
  --output "$ENERGY_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --epochs 200

print ""
print "[4/4] Applying the frozen prospective analysis and decision rule..."
cnn_python studies/cnn_budget_confirmation/analyze.py \
  "$OUTPUT_ROOT" \
  --out "$ANALYSIS_ROOT"

print ""
print "Confirmation run complete."
print "Report:   $ANALYSIS_ROOT/REPORT.md"
print "Decision: $ANALYSIS_ROOT/decision.json"
print "Manifest: $MANIFEST"
