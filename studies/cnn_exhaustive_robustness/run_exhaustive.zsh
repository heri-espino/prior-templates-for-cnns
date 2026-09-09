#!/usr/bin/env zsh
# Run the complete 1200-model exhaustive robustness study without admin rights.
#
# Default workstation-oriented usage:
#   zsh studies/cnn_exhaustive_robustness/run_exhaustive.zsh
#
# Optional environment variables:
#   CNN_ENV_NAME=prior-templates-cnns
#   DEVICE=cuda
#   WORKERS=8
#   THREADS_PER_WORKER=2
#   BATCH_SIZE=256
#   OUTPUT_ROOT=results/exhaustive_robustness_001
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

set -e
set -u
setopt pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
# Keep the default relative. This works both with POSIX Python and with a
# Windows conda.exe invoked from Git-Bash/WSL because the repository is CWD.
OUTPUT_ROOT="${OUTPUT_ROOT:-results/exhaustive_robustness_001}"
CNN_ENV_NAME="${CNN_ENV_NAME:-prior-templates-cnns}"

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
  print -u2 "DEVICE must be cuda or cpu; got: $DEVICE"
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  print -u2 "git is required for provenance recording."
  exit 2
fi

# This helper finds the user's Miniconda/Anaconda installation, creates a named
# user environment if necessary, installs dependencies there, and exposes
# cnn_python. It never invokes an elevated/admin installer.
source "$REPO_ROOT/scripts/use_conda_env.zsh"

TRAIN_ROOT="$OUTPUT_ROOT/training"
EVAL_ROOT="$OUTPUT_ROOT/evaluation"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"
MANIFEST="$OUTPUT_ROOT/execution_manifest.json"
PROTOCOL="studies/cnn_exhaustive_robustness/PROTOCOL.md"

mkdir -p "$OUTPUT_ROOT"

REPO_HEAD="$(git rev-parse HEAD)"
PROTOCOL_COMMIT="$(git log -n 1 --format=%H -- "$PROTOCOL")"
PROTOCOL_GIT_BLOB="$(git hash-object "$PROTOCOL")"

cnn_python - \
  "$MANIFEST" \
  "$REPO_HEAD" \
  "$PROTOCOL_COMMIT" \
  "$PROTOCOL_GIT_BLOB" \
  "$DEVICE" \
  "$WORKERS" \
  "$THREADS_PER_WORKER" \
  "$BATCH_SIZE" <<'PY'
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import psutil
import scipy
import torch

manifest=Path(sys.argv[1])
repo_head,protocol_commit,protocol_git_blob,device,workers,threads,batch=sys.argv[2:]
files=[
    Path('studies/cnn_exhaustive_robustness/PROTOCOL.md'),
    Path('studies/cnn_exhaustive_robustness/train_grid.py'),
    Path('studies/cnn_exhaustive_robustness/evaluate_grid.py'),
    Path('studies/cnn_exhaustive_robustness/analyze_grid.py'),
    Path('studies/cnn_exhaustive_robustness/run_exhaustive.zsh'),
    Path('studies/cnn_release_experiment/core.py'),
    Path('scripts/use_conda_env.zsh'),
]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record={
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'repo_head_at_launch': repo_head,
    'protocol_commit': protocol_commit,
    'protocol_git_blob': protocol_git_blob,
    'protocol_sha256': sha(files[0]),
    'device': device,
    'workers': int(workers),
    'threads_per_worker': int(threads),
    'evaluation_batch_size': int(batch),
    'python': sys.version,
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'cpu_physical': psutil.cpu_count(logical=False),
    'cpu_logical': psutil.cpu_count(logical=True),
    'ram_gib': round(psutil.virtual_memory().total/2**30, 3),
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_vram_gib': round(torch.cuda.get_device_properties(0).total_memory/2**30, 3) if torch.cuda.is_available() else None,
    'source_sha256': {str(p): sha(p) for p in files},
    'frozen_design': {
        'blocks': [5000, 5049],
        'n_blocks': 50,
        'tasks': ['single_shape','two_concepts'],
        'architectures': ['TinyCNN','TwoLayerCNN'],
        'profiles': ['template_init','retention_0p1','retention_1','release_early','release_default','release_late'],
        'epochs': 200,
        'selected_fidelity_budgets': list(range(1,17)),
        'energy_matched_budgets': [1,2,4,8],
        'rankings': ['contrast','auroc','validation_patch'],
        'planned_models': 1200,
    },
}
immutable=['protocol_commit','protocol_sha256','device','threads_per_worker','evaluation_batch_size','source_sha256','frozen_design']
if manifest.exists():
    old=json.loads(manifest.read_text())
    changed=[k for k in immutable if old.get(k)!=record.get(k)]
    if changed:
        raise SystemExit('Existing exhaustive-run manifest is incompatible ('+', '.join(changed)+'). Use a new OUTPUT_ROOT.')
else:
    tmp=manifest.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n')
    tmp.replace(manifest)
print('Execution manifest:',manifest)
print('Protocol commit:',protocol_commit)
print('Conda Python:',sys.executable)
print('CPU:',record['cpu_physical'],'physical /',record['cpu_logical'],'logical')
print('RAM GiB:',record['ram_gib'])
print('GPU:',record['gpu'],'| VRAM GiB:',record['gpu_vram_gib'])
PY

# Avoid accidental BLAS oversubscription in the multiprocessing training grid.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

print ""
print "[1/3] Training the frozen 1200-model robustness grid..."
cnn_python studies/cnn_exhaustive_robustness/train_grid.py \
  --output "$TRAIN_ROOT" \
  --start-block 5000 \
  --blocks 50 \
  --epochs 200 \
  --tasks single_shape two_concepts \
  --architectures TinyCNN TwoLayerCNN \
  --profiles template_init retention_0p1 retention_1 release_early release_default release_late \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS_PER_WORKER"

print ""
print "[2/3] Evaluating k=1..16 and validation-energy-matched controls on the GPU..."
cnn_python studies/cnn_exhaustive_robustness/evaluate_grid.py \
  --input "$TRAIN_ROOT" \
  --output "$EVAL_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE"

print ""
print "[3/3] Applying the frozen robustness analysis..."
cnn_python studies/cnn_exhaustive_robustness/analyze_grid.py \
  "$OUTPUT_ROOT" \
  --out "$ANALYSIS_ROOT"

print ""
print "Exhaustive robustness study complete."
print "Report:   $ANALYSIS_ROOT/REPORT.md"
print "Curves:   $ANALYSIS_ROOT/selected_fidelity_budget_curves.png"
print "Manifest: $MANIFEST"
