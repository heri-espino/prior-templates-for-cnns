# Stage C decomposition of selected versus random fidelity

Generated from the 80 existing Stage C checkpoint evaluations on 9 September 2026 using the updated `review.py`. No model was retrained and no new forward pass was executed. The GitHub Actions run completed successfully and reproduced the stored identity

\[
U=F_{\mathrm{selected}}-F_{\mathrm{random}}
\]

with maximum absolute error 0. The maximum original k4 U discrepancy versus the stored CPU evaluation remains `1.29e-07`; test accuracy discrepancy is 0.

## Main finding

The previously highlighted sign reversal in release-minus-retention `U` is **not equivalent to a sign reversal in selected fidelity**.

For `two_concepts / TinyCNN`, release minus constant retention is:

| Ranking | k | Δ selected fidelity | Δ random fidelity | ΔU | Δ selected CF accuracy |
|---|---:|---:|---:|---:|---:|
| contrast | 1 | -0.375 [-0.464, -0.285] | -0.040 [-0.058, -0.021] | -0.335 [-0.425, -0.245] | -0.255 [-0.333, -0.178] |
| contrast | 2 | -0.163 [-0.267, -0.060] | -0.062 [-0.077, -0.048] | -0.101 [-0.201, -0.001] | -0.185 [-0.320, -0.049] |
| contrast | 4 | -0.022 [-0.054, +0.010] | -0.083 [-0.097, -0.068] | +0.061 [+0.027, +0.095] | -0.030 [-0.070, +0.010] |
| contrast | 8 | +0.006 [+0.004, +0.009] | -0.033 [-0.048, -0.019] | +0.040 [+0.025, +0.055] | +0.013 [+0.007, +0.020] |
| auroc | 1 | -0.332 [-0.435, -0.228] | -0.040 [-0.058, -0.021] | -0.292 [-0.396, -0.189] | -0.230 [-0.316, -0.144] |
| auroc | 2 | -0.188 [-0.292, -0.084] | -0.062 [-0.077, -0.048] | -0.125 [-0.223, -0.028] | -0.218 [-0.344, -0.091] |
| auroc | 4 | -0.022 [-0.058, +0.014] | -0.083 [-0.097, -0.068] | +0.061 [+0.023, +0.098] | -0.031 [-0.079, +0.017] |
| auroc | 8 | +0.007 [+0.004, +0.010] | -0.033 [-0.048, -0.019] | +0.040 [+0.026, +0.055] | +0.015 [+0.009, +0.021] |
| validation_patch | 1 | -0.277 [-0.326, -0.227] | -0.040 [-0.058, -0.021] | -0.237 [-0.283, -0.191] | -0.194 [-0.242, -0.146] |
| validation_patch | 2 | -0.122 [-0.157, -0.086] | -0.062 [-0.077, -0.048] | -0.060 [-0.098, -0.021] | -0.147 [-0.205, -0.089] |
| validation_patch | 4 | -0.015 [-0.033, +0.002] | -0.083 [-0.097, -0.068] | +0.067 [+0.044, +0.090] | -0.021 [-0.046, +0.005] |
| validation_patch | 8 | +0.005 [+0.002, +0.007] | -0.033 [-0.048, -0.019] | +0.038 [+0.023, +0.054] | +0.013 [+0.007, +0.019] |

Thus:

- at `k=1` and `k=2`, release is clearly worse in selected fidelity;
- at `k=4`, selected fidelity is approximately tied/slightly worse, while the random baseline is substantially lower under release; this baseline change creates the positive `ΔU`;
- at `k=8`, selected fidelity itself is slightly higher under release under all three rankings, and selected counterfactual accuracy is also slightly higher.

The most defensible current statement is therefore **budget dependence of the selected-fidelity treatment contrast**, not a simple `U` reversal interpreted as redistribution of causal information.

For `two_concepts / TwoLayerCNN`, the selected-fidelity contrast is negative at k4 under all three rankings (contrast: -0.125; AUROC: -0.121; validation-patch: -0.087), while validation-patch becomes positive at k8 (+0.021 [+0.005, +0.038]). This again supports measurement/budget dependence rather than a fixed architecture-wide sign.

## Manuscript consequence

The current manuscript sentence that emphasizes positive four-channel `U` should be revised after the energy-matched control is available. At minimum, the results must display selected fidelity and random fidelity separately and explicitly state that the k4 TinyCNN `ΔU` advantage is baseline-driven.

The next analysis is the separately frozen validation patch-energy-matched control in `studies/cnn_patch_energy_control/`.