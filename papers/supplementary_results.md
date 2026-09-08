# Supplementary results for the main paper

These tables are part of [the main manuscript](main.md). They reproduce existing analyses without new statistical tests. Values are not selected by significance.

## S1. All prospective primary contrasts

| Setting | ΔU [marginal 95% CI] | Raw p | Holm p | Δalignment | Δaccuracy (pp) |
|---|---:|---:|---:|---:|---:|
| single_shape / TinyCNN | +0.0345 [+0.0018, +0.0672] | 0.0408 | 0.1633 | +0.268 | -0.23 |
| single_shape / TwoLayerCNN | -0.0092 [-0.0513, +0.0329] | 0.6337 | 1.0000 | +0.172 | -0.04 |
| two_concepts / TinyCNN | +0.0252 [-0.0306, +0.0811] | 0.3332 | 0.9995 | +0.289 | -6.13 |
| two_concepts / TwoLayerCNN | -0.0074 [-0.0557, +0.0408] | 0.7358 | 1.0000 | +0.198 | -0.39 |

## S2. All 200-epoch condition means

Mean ± sample SD across ten blocks. All five conditions and all four settings are shown. Accuracy is a percentage; U, alignment, AUROC and IoU are dimensionless.

| Setting | Condition | Accuracy (%) | Alignment | Concept AUROC | Localization IoU | U | Selected CF accuracy (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| single_shape / TinyCNN | random | 100.000 ± 0.000 | 0.300 ± 0.023 | 0.970 ± 0.009 | 0.330 ± 0.063 | 0.249 ± 0.049 | 27.174 ± 6.495 |
| single_shape / TinyCNN | random_unitnorm | 100.000 ± 0.000 | 0.288 ± 0.022 | 0.971 ± 0.011 | 0.340 ± 0.059 | 0.272 ± 0.055 | 30.716 ± 6.203 |
| single_shape / TinyCNN | template_init | 100.000 ± 0.000 | 0.610 ± 0.021 | 0.978 ± 0.006 | 0.409 ± 0.041 | 0.320 ± 0.051 | 35.469 ± 4.792 |
| single_shape / TinyCNN | template_release | 100.000 ± 0.000 | 0.690 ± 0.017 | 0.975 ± 0.007 | 0.438 ± 0.050 | 0.335 ± 0.078 | 38.620 ± 8.816 |
| single_shape / TinyCNN | template_retention_1 | 99.961 ± 0.124 | 0.959 ± 0.003 | 0.938 ± 0.008 | 0.544 ± 0.037 | 0.358 ± 0.048 | 50.169 ± 5.142 |
| single_shape / TwoLayerCNN | random | 100.000 ± 0.000 | 0.261 ± 0.037 | 0.830 ± 0.025 | 0.307 ± 0.050 | 0.014 ± 0.024 | 2.799 ± 2.644 |
| single_shape / TwoLayerCNN | random_unitnorm | 100.000 ± 0.000 | 0.228 ± 0.033 | 0.838 ± 0.019 | 0.308 ± 0.060 | 0.004 ± 0.032 | 2.279 ± 1.985 |
| single_shape / TwoLayerCNN | template_init | 100.000 ± 0.000 | 0.799 ± 0.015 | 0.840 ± 0.016 | 0.470 ± 0.024 | 0.056 ± 0.061 | 5.924 ± 6.345 |
| single_shape / TwoLayerCNN | template_release | 99.961 ± 0.124 | 0.942 ± 0.006 | 0.846 ± 0.016 | 0.504 ± 0.039 | 0.060 ± 0.065 | 7.552 ± 7.152 |
| single_shape / TwoLayerCNN | template_retention_1 | 99.922 ± 0.165 | 1.000 ± 0.000 | 0.874 ± 0.011 | 0.566 ± 0.043 | 0.067 ± 0.065 | 9.961 ± 10.034 |
| two_concepts / TinyCNN | random | 99.258 ± 0.892 | 0.232 ± 0.034 | 0.988 ± 0.009 | 0.180 ± 0.030 | 0.632 ± 0.060 | 73.906 ± 9.590 |
| two_concepts / TinyCNN | random_unitnorm | 99.297 ± 0.708 | 0.222 ± 0.031 | 0.986 ± 0.009 | 0.188 ± 0.030 | 0.607 ± 0.070 | 71.797 ± 10.205 |
| two_concepts / TinyCNN | template_init | 99.297 ± 0.514 | 0.507 ± 0.022 | 0.987 ± 0.005 | 0.214 ± 0.034 | 0.629 ± 0.065 | 77.949 ± 10.461 |
| two_concepts / TinyCNN | template_release | 99.219 ± 0.689 | 0.580 ± 0.011 | 0.984 ± 0.006 | 0.235 ± 0.041 | 0.718 ± 0.079 | 90.645 ± 6.945 |
| two_concepts / TinyCNN | template_retention_1 | 98.008 ± 0.982 | 0.947 ± 0.004 | 0.946 ± 0.012 | 0.347 ± 0.017 | 0.657 ± 0.056 | 93.613 ± 3.604 |
| two_concepts / TwoLayerCNN | random | 99.492 ± 0.522 | 0.255 ± 0.027 | 0.806 ± 0.042 | 0.223 ± 0.027 | 0.066 ± 0.058 | 8.145 ± 6.561 |
| two_concepts / TwoLayerCNN | random_unitnorm | 99.375 ± 0.906 | 0.243 ± 0.028 | 0.785 ± 0.033 | 0.218 ± 0.013 | 0.069 ± 0.065 | 8.438 ± 5.469 |
| two_concepts / TwoLayerCNN | template_init | 99.570 ± 0.430 | 0.746 ± 0.017 | 0.785 ± 0.039 | 0.326 ± 0.015 | 0.007 ± 0.048 | 4.453 ± 4.286 |
| two_concepts / TwoLayerCNN | template_release | 99.531 ± 0.546 | 0.907 ± 0.012 | 0.771 ± 0.036 | 0.363 ± 0.018 | 0.003 ± 0.040 | 4.023 ± 3.412 |
| two_concepts / TwoLayerCNN | template_retention_1 | 99.141 ± 0.576 | 0.999 ± 0.000 | 0.776 ± 0.010 | 0.391 ± 0.014 | 0.061 ± 0.052 | 8.242 ± 3.847 |

## S3. Complete measurement-robustness contrasts

Release minus constant retention. All 96 contrasts: 4 settings × 3 rankings × 4 sizes × 2 outcomes. Intervals are marginal 95% paired t intervals; no multiplicity adjustment. Neither exclusion of zero nor agreement of several dependent intervals is a confirmatory discovery. CF accuracy differences are shown as proportions.

### causal_usefulness

| Setting | Ranking | k | Mean difference | 95% interval |
|---|---|---:|---:|---:|
| single_shape / TinyCNN | auroc | 1 | -0.0764 | [-0.0984, -0.0544] |
| single_shape / TinyCNN | auroc | 2 | -0.1071 | [-0.1337, -0.0805] |
| single_shape / TinyCNN | auroc | 4 | -0.0309 | [-0.0774, +0.0155] |
| single_shape / TinyCNN | auroc | 8 | +0.0255 | [-0.0057, +0.0567] |
| single_shape / TinyCNN | contrast | 1 | -0.0616 | [-0.0852, -0.0380] |
| single_shape / TinyCNN | contrast | 2 | -0.1252 | [-0.1661, -0.0843] |
| single_shape / TinyCNN | contrast | 4 | -0.0224 | [-0.0874, +0.0426] |
| single_shape / TinyCNN | contrast | 8 | +0.0312 | [+0.0042, +0.0583] |
| single_shape / TinyCNN | validation_patch | 1 | -0.0955 | [-0.1144, -0.0767] |
| single_shape / TinyCNN | validation_patch | 2 | -0.0590 | [-0.0991, -0.0189] |
| single_shape / TinyCNN | validation_patch | 4 | +0.0240 | [-0.0301, +0.0782] |
| single_shape / TinyCNN | validation_patch | 8 | +0.0430 | [+0.0156, +0.0704] |
| single_shape / TwoLayerCNN | auroc | 1 | +0.0019 | [-0.0001, +0.0039] |
| single_shape / TwoLayerCNN | auroc | 2 | +0.0011 | [-0.0054, +0.0076] |
| single_shape / TwoLayerCNN | auroc | 4 | -0.0139 | [-0.0668, +0.0391] |
| single_shape / TwoLayerCNN | auroc | 8 | +0.0528 | [-0.0067, +0.1123] |
| single_shape / TwoLayerCNN | contrast | 1 | +0.0016 | [-0.0003, +0.0035] |
| single_shape / TwoLayerCNN | contrast | 2 | -0.0015 | [-0.0090, +0.0060] |
| single_shape / TwoLayerCNN | contrast | 4 | -0.0066 | [-0.0521, +0.0388] |
| single_shape / TwoLayerCNN | contrast | 8 | +0.0503 | [-0.0042, +0.1047] |
| single_shape / TwoLayerCNN | validation_patch | 1 | -0.0126 | [-0.0182, -0.0070] |
| single_shape / TwoLayerCNN | validation_patch | 2 | -0.0170 | [-0.0346, +0.0005] |
| single_shape / TwoLayerCNN | validation_patch | 4 | +0.0017 | [-0.0530, +0.0564] |
| single_shape / TwoLayerCNN | validation_patch | 8 | +0.0920 | [+0.0437, +0.1402] |
| two_concepts / TinyCNN | auroc | 1 | -0.2923 | [-0.3960, -0.1886] |
| two_concepts / TinyCNN | auroc | 2 | -0.1254 | [-0.2228, -0.0279] |
| two_concepts / TinyCNN | auroc | 4 | +0.0606 | [+0.0230, +0.0982] |
| two_concepts / TinyCNN | auroc | 8 | +0.0405 | [+0.0259, +0.0551] |
| two_concepts / TinyCNN | contrast | 1 | -0.3352 | [-0.4250, -0.2453] |
| two_concepts / TinyCNN | contrast | 2 | -0.1011 | [-0.2007, -0.0015] |
| two_concepts / TinyCNN | contrast | 4 | +0.0608 | [+0.0270, +0.0947] |
| two_concepts / TinyCNN | contrast | 8 | +0.0398 | [+0.0248, +0.0548] |
| two_concepts / TinyCNN | validation_patch | 1 | -0.2371 | [-0.2832, -0.1909] |
| two_concepts / TinyCNN | validation_patch | 2 | -0.0595 | [-0.0978, -0.0213] |
| two_concepts / TinyCNN | validation_patch | 4 | +0.0672 | [+0.0440, +0.0903] |
| two_concepts / TinyCNN | validation_patch | 8 | +0.0383 | [+0.0229, +0.0538] |
| two_concepts / TwoLayerCNN | auroc | 1 | -0.0028 | [-0.0101, +0.0045] |
| two_concepts / TwoLayerCNN | auroc | 2 | +0.0034 | [-0.0093, +0.0160] |
| two_concepts / TwoLayerCNN | auroc | 4 | -0.0539 | [-0.0924, -0.0153] |
| two_concepts / TwoLayerCNN | auroc | 8 | +0.0109 | [-0.1079, +0.1297] |
| two_concepts / TwoLayerCNN | contrast | 1 | -0.0010 | [-0.0084, +0.0063] |
| two_concepts / TwoLayerCNN | contrast | 2 | -0.0000 | [-0.0152, +0.0152] |
| two_concepts / TwoLayerCNN | contrast | 4 | -0.0578 | [-0.0944, -0.0212] |
| two_concepts / TwoLayerCNN | contrast | 8 | -0.0038 | [-0.0964, +0.0889] |
| two_concepts / TwoLayerCNN | validation_patch | 1 | -0.0129 | [-0.0191, -0.0066] |
| two_concepts / TwoLayerCNN | validation_patch | 2 | -0.0433 | [-0.0640, -0.0227] |
| two_concepts / TwoLayerCNN | validation_patch | 4 | -0.0206 | [-0.0954, +0.0542] |
| two_concepts / TwoLayerCNN | validation_patch | 8 | +0.0429 | [+0.0123, +0.0735] |

### cf_accuracy

| Setting | Ranking | k | Mean difference | 95% interval |
|---|---|---:|---:|---:|
| single_shape / TinyCNN | auroc | 1 | -0.0255 | [-0.0438, -0.0073] |
| single_shape / TinyCNN | auroc | 2 | -0.0784 | [-0.1011, -0.0557] |
| single_shape / TinyCNN | auroc | 4 | -0.1272 | [-0.1782, -0.0762] |
| single_shape / TinyCNN | auroc | 8 | -0.0220 | [-0.0601, +0.0161] |
| single_shape / TinyCNN | contrast | 1 | -0.0163 | [-0.0322, -0.0003] |
| single_shape / TinyCNN | contrast | 2 | -0.0803 | [-0.1181, -0.0426] |
| single_shape / TinyCNN | contrast | 4 | -0.1155 | [-0.1782, -0.0528] |
| single_shape / TinyCNN | contrast | 8 | -0.0122 | [-0.0442, +0.0197] |
| single_shape / TinyCNN | validation_patch | 1 | -0.0383 | [-0.0583, -0.0183] |
| single_shape / TinyCNN | validation_patch | 2 | -0.0691 | [-0.1066, -0.0317] |
| single_shape / TinyCNN | validation_patch | 4 | -0.0404 | [-0.1083, +0.0275] |
| single_shape / TinyCNN | validation_patch | 8 | -0.0000 | [-0.0312, +0.0312] |
| single_shape / TwoLayerCNN | auroc | 1 | -0.0005 | [-0.0017, +0.0007] |
| single_shape / TwoLayerCNN | auroc | 2 | -0.0007 | [-0.0019, +0.0006] |
| single_shape / TwoLayerCNN | auroc | 4 | -0.0216 | [-0.0818, +0.0386] |
| single_shape / TwoLayerCNN | auroc | 8 | +0.0439 | [-0.0217, +0.1095] |
| single_shape / TwoLayerCNN | contrast | 1 | -0.0005 | [-0.0017, +0.0007] |
| single_shape / TwoLayerCNN | contrast | 2 | -0.0012 | [-0.0027, +0.0004] |
| single_shape / TwoLayerCNN | contrast | 4 | -0.0241 | [-0.0841, +0.0359] |
| single_shape / TwoLayerCNN | contrast | 8 | +0.0402 | [-0.0295, +0.1099] |
| single_shape / TwoLayerCNN | validation_patch | 1 | -0.0012 | [-0.0032, +0.0009] |
| single_shape / TwoLayerCNN | validation_patch | 2 | -0.0031 | [-0.0109, +0.0047] |
| single_shape / TwoLayerCNN | validation_patch | 4 | -0.0112 | [-0.0721, +0.0497] |
| single_shape / TwoLayerCNN | validation_patch | 8 | +0.1203 | [+0.0805, +0.1602] |
| two_concepts / TinyCNN | auroc | 1 | -0.2301 | [-0.3162, -0.1440] |
| two_concepts / TinyCNN | auroc | 2 | -0.2176 | [-0.3442, -0.0910] |
| two_concepts / TinyCNN | auroc | 4 | -0.0311 | [-0.0790, +0.0169] |
| two_concepts / TinyCNN | auroc | 8 | +0.0146 | [+0.0087, +0.0205] |
| two_concepts / TinyCNN | contrast | 1 | -0.2555 | [-0.3332, -0.1777] |
| two_concepts / TinyCNN | contrast | 2 | -0.1846 | [-0.3204, -0.0487] |
| two_concepts / TinyCNN | contrast | 4 | -0.0297 | [-0.0695, +0.0101] |
| two_concepts / TinyCNN | contrast | 8 | +0.0133 | [+0.0070, +0.0195] |
| two_concepts / TinyCNN | validation_patch | 1 | -0.1939 | [-0.2421, -0.1458] |
| two_concepts / TinyCNN | validation_patch | 2 | -0.1471 | [-0.2049, -0.0892] |
| two_concepts / TinyCNN | validation_patch | 4 | -0.0205 | [-0.0461, +0.0051] |
| two_concepts / TinyCNN | validation_patch | 8 | +0.0129 | [+0.0068, +0.0190] |
| two_concepts / TwoLayerCNN | auroc | 1 | -0.0027 | [-0.0062, +0.0008] |
| two_concepts / TwoLayerCNN | auroc | 2 | -0.0027 | [-0.0085, +0.0031] |
| two_concepts / TwoLayerCNN | auroc | 4 | -0.0406 | [-0.0652, -0.0161] |
| two_concepts / TwoLayerCNN | auroc | 8 | +0.0064 | [-0.1208, +0.1337] |
| two_concepts / TwoLayerCNN | contrast | 1 | -0.0029 | [-0.0064, +0.0005] |
| two_concepts / TwoLayerCNN | contrast | 2 | -0.0031 | [-0.0096, +0.0033] |
| two_concepts / TwoLayerCNN | contrast | 4 | -0.0422 | [-0.0659, -0.0185] |
| two_concepts / TwoLayerCNN | contrast | 8 | -0.0064 | [-0.1039, +0.0910] |
| two_concepts / TwoLayerCNN | validation_patch | 1 | -0.0033 | [-0.0066, -0.0001] |
| two_concepts / TwoLayerCNN | validation_patch | 2 | -0.0113 | [-0.0247, +0.0020] |
| two_concepts / TwoLayerCNN | validation_patch | 4 | -0.0539 | [-0.1222, +0.0144] |
| two_concepts / TwoLayerCNN | validation_patch | 8 | +0.0305 | [+0.0073, +0.0536] |

## S4. Data and analysis map

- Prospective raw per-run results: `studies/cnn_causal_milestone/analysis/per_run.csv`.
- Retention-release: `analysis/retention_release_001/endpoints.csv` and `paired_contrasts.csv` (all 160 exploratory endpoint contrasts).
- Robustness raw method/size data: `results/patch_robustness_gpu_001/analysis/metrics.csv`.
- Rebuild these tables and the full sensitivity figure with `python analysis/main_paper/build_assets.py`.
- This script checks completeness and presentation, not saved model forwards. The final independent reproducibility review is deferred.
