#!/usr/bin/env zsh
# Run the frozen prospective intervention-budget confirmation end to end.
#
# Usage from anywhere inside/outside the repository:
#   zsh studies/cnn_budget_confirmation/run_confirmation.zsh
#
# Optional environment variables:
#   PYTHON_BIN=python3
#   OUTPUT_ROOT=/absolute/or/relative/path
#   DEVICE=cuda                 # cuda or cpu, evaluation only
#   BATCH_SIZE=64
#   THREADS=2                   # deterministic Stage-B-style CPU training
#   INSTALL_DEPS=0              # set to 1 to pip-install the frozen requirements
#
# The trainer and evaluators are resumable. Re-running this script with the same
# OUTPUT_ROOT resumes/skips completed work under the same frozen design.

set -e
set -u
setopt pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/budget_confirmation_001}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THREADS="${THREADS:-2}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

TRAIN_ROOT="$OUTPUT_ROOT/training"
PATCH_ROOT="$OUTPUT_ROOT/patch_eval"
ENERGY_ROOT="$OUTPUT_ROOT/energy_eval"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis/budget_confirmation"
MANIFEST="$OUTPUT_ROOT/execution_manifest.json"
PROTOCOL="studies/cnn_budget_confirmation/PROTOCOL.md"

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
  print -u2 "DEVICE must be 'cuda' or 'cpu'; got: $DEVICE"
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  print -u2 "Python executable not found: $PYTHON_BIN"
  exit 2
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r studies/cnn_release_experiment/requirements.txt
fi

# Fail before producing planned outcomes when required imports / CUDA are absent.
"$PYTHON_BIN" - "$DEVICE" <<'PY'
import sys
import numpy
import pandas
import scipy
import torch
from PIL import Image

device=sys.argv[1]
if device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit(
        'DEVICE=cuda but torch.cuda.is_available() is False. '
        'Install a CUDA-enabled PyTorch build or rerun with DEVICE=cpu.'
    )
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$OUTPUT_ROOT"

# Record the exact frozen protocol and code state before training starts. The
# protocol commit is the commit that last changed PROTOCOL.md, not current HEAD.
REPO_HEAD="$(git rev-parse HEAD)"
PROTOCOL_COMMIT="$(git log -n 1 --format=%H -- "$PROTOCOL")"
PROTOCOL_SHA256="$(git hash-object "$PROTOCOL")"

"$PYTHON_BIN" - "$MANIFEST" "$REPO_HEAD" "$PROTOCOL_COMMIT" "$PROTOCOL_SHA256" "$DEVICE" "$BATCH_SIZE" "$THREADS" <<'PY'
import hashlib
import json
import platform
import subprocess
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
        raise SystemExit(
            'Existing execution manifest is incompatible with this run ('
            + ', '.join(changed)
            + '). Use a new OUTPUT_ROOT instead of mixing designs/code states.'
        )
else:
    tmp=manifest.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n')
    tmp.replace(manifest)
print('Execution manifest:', manifest)
print('Protocol commit:', protocol_commit)
print('Protocol SHA-256:', record['protocol_sha256'])
PY

print ""
print "[1/4] Training 80 fresh models on blocks 4000-4019..."
"$PYTHON_BIN" studies/cnn_release_experiment/experiment.py \
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
"$PYTHON_BIN" studies/cnn_patch_robustness/evaluate_gpu.py \
  --input "$TRAIN_ROOT" \
  --output "$PATCH_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --epochs 200

print ""
print "[3/4] Evaluating validation-energy-matched controls..."
"$PYTHON_BIN" studies/cnn_patch_energy_control/evaluate_gpu.py \
  --input "$TRAIN_ROOT" \
  --output "$ENERGY_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --epochs 200

print ""
print "[4/4] Applying the frozen prospective analysis and decision rule..."
"$PYTHON_BIN" studies/cnn_budget_confirmation/analyze.py \
  "$OUTPUT_ROOT" \
  --out "$ANALYSIS_ROOT"

print ""
print "Confirmation run complete."
print "Report:   $ANALYSIS_ROOT/REPORT.md"
print "Decision: $ANALYSIS_ROOT/decision.json"
print "Manifest: $MANIFEST"
