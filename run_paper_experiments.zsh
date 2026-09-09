#!/usr/bin/env zsh
# Run the paper's frozen prospective confirmation followed by the separate
# exhaustive robustness study. Both child launchers create/use the same
# user-space Conda environment and require no administrator privileges.
#
# Usage:
#   zsh run_paper_experiments.zsh
#
# Common optional overrides:
#   CNN_ENV_NAME=prior-templates-cnns
#   DEVICE=cuda
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#
# Exhaustive-stage compute overrides:
#   WORKERS=8 THREADS_PER_WORKER=2 BATCH_SIZE=256

set -e
set -u
setopt pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

print "=== Stage D: frozen prospective confirmation (blocks 4000-4019) ==="
zsh studies/cnn_budget_confirmation/run_confirmation.zsh

print ""
print "=== Stage E: separate exhaustive robustness map (blocks 5000-5049) ==="
zsh studies/cnn_exhaustive_robustness/run_exhaustive.zsh

print ""
print "All planned paper experiments finished."
print "Primary confirmation report: results/budget_confirmation_001/analysis/budget_confirmation/REPORT.md"
print "Exhaustive robustness report: results/exhaustive_robustness_001/analysis/REPORT.md"
