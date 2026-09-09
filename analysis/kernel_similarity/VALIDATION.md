# Validation and scope

- Five known-answer metric tests pass: permutation/positive scale/centering invariance, duplicate-template pressure, signed polarity, undefined zero-norm rejection, and agreement with exhaustive assignment on a four-kernel example.
- The shell launcher completed on macOS with CPU PyTorch 2.8.0, NumPy 2.0.2 and SciPy 1.13.1. The PowerShell launcher was inspected but not executed on this host.
- All 200 Stage B models and 2,000 checkpoints were read with the restricted PyTorch weights-only loader. No unsafe fallback, model inference or retraining was used.
- All 1,800 saved post-training alignment values matched recomputation exactly in this execution.
- Checked 2,000 finite 16×16 matrices, 32,000 channel rows, the row-max aggregate, and unique template assignments for every checkpoint. Each matching gap is nonnegative up to floating-point tolerance.
- Preserve the original experimental files. This analysis cannot validate the renderer, saved behavior metrics or causal fidelity by independent forward evaluation; it joins existing results.

The implementation computes small dot products by explicit elementwise sums, matching the preserved core and avoiding platform-specific matrix-multiplication warnings encountered during plotting on this host. No warning suppression or nonfinite-value replacement is used.

Paper additions are exploratory. No correlation p-values, selected significant subgroups, human-readability claims or submission-audit sign-off were added.
