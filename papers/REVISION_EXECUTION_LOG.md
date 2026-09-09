# Revision execution log

Date: 2026-09-09
Branch: `paper_suggestions`

This log records changes made after the scientific review. It is not a substitute for the protocols or generated analysis reports.

## Completed

### 0. Staged revision plan

Added `REVISION_PLAN.md` with explicit separation among exploratory reanalysis, checkpoint robustness controls, independent audit, future prospective confirmation, and submission packaging.

### 1. Decompose `U`

Updated `analysis/patch_robustness_gpu_001/review.py` to expose:

- selected fidelity;
- random fidelity;
- `U`;
- selected and random counterfactual accuracy;
- paired release-minus-retention intervals for each component.

A GitHub Actions run regenerated the analysis successfully from all 80 existing Stage C checkpoint evaluations.

Main result on `two_concepts / TinyCNN`:

- selected fidelity is strongly worse under release at k1/k2;
- at k4 selected fidelity is approximately tied/slightly worse, while random fidelity is much lower under release, creating the positive ΔU;
- at k8 selected fidelity itself is slightly higher under release under all three rankings.

Therefore the manuscript must not interpret the positive k4 ΔU as direct evidence that the selected release channels are functionally better. See `analysis/patch_robustness_gpu_001/STAGE1_DECOMPOSITION.md`.

### 2. Validation patch-energy-matched control — protocol and implementation

Frozen `studies/cnn_patch_energy_control/PROTOCOL.md` before generating matched-control outcomes.

Implemented:

- exhaustive same-k subset matching to validation activation replacement energy;
- eight closest controls per concept and ranking;
- unchanged Stage C random baseline for continuity;
- selected, unmatched-random and energy-matched fidelity/CF-accuracy outputs;
- matching diagnostics and Stage C consistency checks;
- resumable runner and summary code.

The CI execution of this post-hoc checkpoint control is tracked separately from the original Stage C evidence.

### 4. Independent checkpoint audit — protocol and implementation

Frozen `studies/cnn_checkpoint_audit/AUDIT_PROTOCOL.md`.

Implemented a separate metric computation path that does not call the original Stage B evaluator for:

- test accuracy;
- template alignment;
- concept AUROC;
- localization IoU;
- selected/random patch fidelity;
- U;
- counterfactual accuracy;
- full/no-op patch identities.

Planned sample: 32 checkpoints spanning both tasks, architectures, treatments, blocks 2000/2009 and epochs 80/200.

## Pending numerical review in the current revision

- energy-matched control output;
- independent audit output;
- manuscript rewrite based on those results;
- final prospective confirmation protocol;
- fresh confirmation execution;
- anonymous submission artifact.

No pending result is assumed to have a favorable sign.