# Stage 2 validation patch-energy-matched control

Date: 2026-09-09
Status: exploratory robustness analysis of the existing Stage B checkpoints/test data.

All **80/80** planned checkpoint evaluations completed. The evaluator also recomputed the unchanged Stage C path; maximum absolute discrepancies were `2.38e-7` for selected fidelity, `1.75e-7` for random fidelity, `1.78e-7` for the original U, and `5.96e-8` for selected counterfactual accuracy.

## What the control changes

The original baseline chooses same-size random channel sets. The new baseline chooses, using validation activations only, eight alternative same-size channel sets whose first-layer activation replacement energy is closest to the selected set. This yields

\[
U_{\mathrm{energy}}=F_{\mathrm{selected}}-F_{\mathrm{energy\ matched}}.
\]

This is a robustness diagnostic, not a new independent experiment.

## Matching quality

Matching quality depends strongly on channel budget.

- `k=1` is often poorly matchable because there may be no alternative singleton channel with comparable replacement energy. On `two_concepts/TinyCNN/template_release`, maximum relative mismatch reaches 2.84. The k1 energy-matched baseline must therefore be interpreted cautiously.
- `k=2` improves substantially but some settings still show material mismatch.
- `k=4` and `k=8` are generally much tighter. For `two_concepts/TinyCNN/template_release`, mean relative mismatch is about 0.0012 at k4 and 0.0001 at k8 across rankings. For retained TinyCNN, k4 mean mismatch ranges about 0.0035–0.0065 and k8 about 0.0001–0.0002.
- TwoLayerCNN k4 is tight for validation-patch ranking (mean mismatch about 0.0013 retained and 0.0033 release), but contrast/AUROC retained sets are less perfectly matchable (mean mismatch about 0.055, maximum about 0.209). k8 is again very tight.

The matched-control result should therefore be displayed together with matching quality, not labeled as uniformly matched at every k.

## Main scientific result

The Stage 1 conclusion survives: **selected fidelity itself is strongly intervention-budget dependent**, independently of how U is baselined.

### `two_concepts / TinyCNN`

For release minus constant retention, selected fidelity is:

- strongly negative at k1 and k2 under all three validation-only rankings;
- near zero/slightly negative at k4 (all marginal intervals include zero);
- slightly positive at k8 under all three rankings, with marginal intervals above zero.

The original positive k4 `ΔU_random` is baseline-driven: selected fidelity is not positive there. The energy-matched relative score is also positive at k4 for contrast/AUROC and has a positive mean for validation-patch, but this must not be used to overwrite the more direct selected-fidelity result.

Selected-fidelity contrasts at k8 are:

- contrast: `+0.0064 [+0.0036, +0.0093]`;
- AUROC: `+0.0071 [+0.0041, +0.0102]`;
- validation-patch: `+0.0049 [+0.0025, +0.0074]`.

Selected counterfactual accuracy follows the same broad pattern: negative at small budgets, approximately tied at k4, and slightly positive at k8.

### `two_concepts / TwoLayerCNN`

The deeper model does not have one stable treatment sign either.

- contrast/AUROC selected fidelity is negative through k4 and uncertain at k8;
- validation-patch selected fidelity is negative at k1, k2 and k4, but positive at k8: `+0.0215 [+0.0049, +0.0381]`;
- the energy-matched relative score is negative at validation-patch k4 (`-0.0717 [-0.1413, -0.0020]`) and uncertain at k8.

Thus the original simple architecture story—release helps shallow causal usefulness but harms deeper causal usefulness—is not supported as an invariant statement.

## Manuscript consequence

The main paper should make **selected fidelity** and **absolute selected-patch counterfactual accuracy** the primary Stage C robustness displays. `U_random` should remain as the original relative metric, with its selected/random decomposition shown. `U_energy` should be reported as a secondary post-hoc control with matching-quality diagnostics.

The defensible claim is:

> Within the tested compositional task, the release-minus-retention difference in selected-channel intervention fidelity changes substantially with intervention budget. In TinyCNN it is clearly negative for one or two patched channels, approximately tied at four, and slightly positive at eight across three validation-only rankings. TwoLayerCNN also exhibits ranking- and budget-dependent treatment differences. Relative selected-versus-control scores additionally depend on the baseline definition.

Do **not** claim that release has been shown to redistribute causal information across channels. The current analyses establish dependence of the measured treatment comparison on intervention budget and baseline, not the underlying mechanism.