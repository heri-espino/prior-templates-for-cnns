# Prospective confirmation: intervention-budget dependence of selected fidelity

Protocol frozen: 2026-09-09
Branch: `paper_suggestions`

## Status and motivation

This protocol is written **after** the Stage B/C exploratory results, the decomposition of `U`, and the validation patch-energy-matched control are known, but **before any block 4000–4019 training or evaluation outcomes are generated**.

The discovery result that survives the robustness review is not the original sign reversal of

\[
U=F_{\mathrm{selected}}-F_{\mathrm{random}},
\]

because the positive k4 TinyCNN `ΔU` is substantially baseline-driven. The narrower result to confirm is that the **release-minus-retention treatment difference in selected-channel fidelity changes with intervention budget**.

This confirmation therefore makes selected fidelity primary. Relative selected-versus-control scores remain secondary diagnostics.

## Fresh experimental sample

Use new deterministic renderer blocks **4000–4019**, which were not used in Stages A–C.

Training design:

- task: `two_concepts` only;
- architectures: `TinyCNN`, `TwoLayerCNN`;
- treatments: `template_retention_1`, `template_release`;
- 20 independent blocks per architecture;
- 200 epochs;
- same renderer distribution, optimizer, learning rate, batch size and release schedule as Stage B;
- final epoch 200 is the inferential checkpoint;
- no early stopping or outcome-based checkpoint choice.

Total training runs: 80 models. TinyCNN supplies the single primary confirmation test; TwoLayerCNN is a predeclared secondary generalization analysis.

The sample size is fixed at 20 blocks—twice the exploratory Stage B block count—before confirmation outcomes are produced. It is not changed based on interim results.

## Validation-only rankings and intervention budgets

At epoch 200 evaluate the same three rankings:

1. `contrast`: standardized positive–negative first-layer activation contrast;
2. `auroc`: absolute deviation of validation concept AUROC from 0.5;
3. `validation_patch`: singleton validation counterfactual fidelity.

Evaluate all rankings at

\[
k\in\{1,2,4,8\}.
\]

No test output is used for channel selection or for choosing k.

## Primary estimand and test

The **single primary ranking** is `contrast`, because it is the original Stage B/C ranking and is observational with respect to the downstream patching metric.

For block `b` and intervention size `k`, define the paired treatment effect

\[
\Delta_b(k)=F_{\mathrm{selected},b}^{\mathrm{release}}(k)-F_{\mathrm{selected},b}^{\mathrm{retention}}(k).
\]

Define the pre-specified budget contrast

\[
B_b=\frac{\Delta_b(4)+\Delta_b(8)}{2}
    -\frac{\Delta_b(1)+\Delta_b(2)}{2}.
\]

`B > 0` means release becomes relatively more favorable (or less unfavorable) when moving from small to larger interventions.

### Primary hypothesis

For `two_concepts / TinyCNN / contrast`, test

\[
H_0:\;\mathbb E[B]=0
\]

with a two-sided one-sample Student t test over the 20 block values. Report the mean, sample SD, 95% t interval, t statistic and p value.

The discovery is considered independently confirmed only if:

1. all 20 planned blocks complete;
2. the mean `B` is positive;
3. its two-sided 95% t interval excludes zero (equivalently p < 0.05 under this single test);
4. the four individual `Δ(k)` estimates are all reported, regardless of sign.

There is no multiplicity correction for the primary claim because there is exactly one primary test.

## Predeclared secondary analyses

These do not alter the primary decision:

- the same `B` contrast for TinyCNN under `auroc` and `validation_patch`;
- the same three rankings and full k curve for TwoLayerCNN;
- absolute selected-patch counterfactual accuracy at every k;
- original same-size random baseline and `U_random`;
- validation patch-energy-matched baseline and `U_energy`;
- energy-matching diagnostics;
- ordinary test accuracy and full/no-op patch identities.

Secondary intervals are marginal and exploratory unless an explicit correction is stated in the generated report.

## Energy-matched control

Use the already frozen Stage 2 procedure without alteration: for each concept/ranking/k, select eight same-k alternative channel sets whose validation first-layer replacement energy is closest to the selected set. Test data are evaluated only after these controls are fixed.

Poorly matchable settings—especially singleton k=1—must retain their matching-error diagnostics and may not be described as successfully matched by definition.

## Integrity and stopping rules

- No outcome from blocks 4000–4019 may be inspected before this protocol commit exists.
- All 80 planned training runs are retained; failed runs are debugged/restarted under the same design rather than dropped because of outcome.
- The final checkpoint is epoch 200 for every run.
- No architecture, ranking or k is removed after seeing outcomes.
- Full 16-channel and zero-channel patch identities must pass the existing tolerances.
- Source hashes, environment, checkpoint hashes and result files are retained.
- The confirmation report must state the exact protocol commit/hash.

## Interpretation boundary

A positive primary `B` confirms intervention-budget dependence of the **measured selected-fidelity treatment contrast within this renderer/model setting**. It does not prove that release redistributes causal information, identify a unique mechanism, establish human interpretability, or generalize to natural images.

Failure of the primary test narrows the paper: the original Stage C result remains a transparent exploratory sensitivity finding, but is not promoted as independently replicated evidence.