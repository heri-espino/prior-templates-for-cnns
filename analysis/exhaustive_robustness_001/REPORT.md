# Exhaustive template-prior robustness study

Statistical status: secondary robustness study. It does **not** alter the frozen 20-block prospective confirmation.

Completed final-checkpoint evaluations: **1200 / 1200**.
Fresh renderer blocks: **5000--5049**.

## Default-release replication across settings

The table reports the predeclared selected-fidelity budget contrast B for `release_default - retention_1` under the contrast ranking.

| Task | Architecture | mean B | 95% interval | p | Holm p within setting |
|---|---|---:|---:|---:|---:|
| single_shape | TinyCNN | +0.07189 | [+0.05444, +0.08934] | 7.154e-11 | 1.431e-10 |
| single_shape | TwoLayerCNN | -0.01018 | [-0.02290, +0.00254] | 0.1142 | 0.4569 |
| two_concepts | TinyCNN | +0.28619 | [+0.25389, +0.31850] | 4.883e-23 | 1.953e-22 |
| two_concepts | TwoLayerCNN | -0.04156 | [-0.06983, -0.01329] | 0.0048 | 0.0144 |

## Full profile map (contrast ranking)

| Task | Architecture | Profile vs retention_1 | mean B | 95% interval | Holm p |
|---|---|---|---:|---:|---:|
| single_shape | TinyCNN | template_init | +0.06804 | [+0.05343, +0.08264] | 6.914e-12 |
| single_shape | TinyCNN | retention_0p1 | +0.03871 | [+0.02525, +0.05218] | 5.134e-07 |
| single_shape | TinyCNN | release_early | +0.06855 | [+0.05228, +0.08482] | 1.111e-10 |
| single_shape | TinyCNN | release_default | +0.07189 | [+0.05444, +0.08934] | 1.431e-10 |
| single_shape | TinyCNN | release_late | +0.07955 | [+0.06404, +0.09505] | 3.612e-13 |
| single_shape | TwoLayerCNN | template_init | +0.00762 | [-0.00793, +0.02317] | 0.9885 |
| single_shape | TwoLayerCNN | retention_0p1 | +0.00031 | [-0.01053, +0.01116] | 1 |
| single_shape | TwoLayerCNN | release_early | +0.00003 | [-0.01500, +0.01506] | 1 |
| single_shape | TwoLayerCNN | release_default | -0.01018 | [-0.02290, +0.00254] | 0.4569 |
| single_shape | TwoLayerCNN | release_late | -0.01237 | [-0.02194, -0.00279] | 0.06212 |
| two_concepts | TinyCNN | template_init | +0.34557 | [+0.31797, +0.37317] | 5.607e-29 |
| two_concepts | TinyCNN | retention_0p1 | +0.16626 | [+0.14002, +0.19251] | 3.713e-17 |
| two_concepts | TinyCNN | release_early | +0.32245 | [+0.28415, +0.36074] | 1.265e-21 |
| two_concepts | TinyCNN | release_default | +0.28619 | [+0.25389, +0.31850] | 1.953e-22 |
| two_concepts | TinyCNN | release_late | +0.19342 | [+0.16432, +0.22253] | 1.197e-17 |
| two_concepts | TwoLayerCNN | template_init | -0.00583 | [-0.03214, +0.02049] | 0.6583 |
| two_concepts | TwoLayerCNN | retention_0p1 | -0.01413 | [-0.03880, +0.01054] | 0.5104 |
| two_concepts | TwoLayerCNN | release_early | -0.04199 | [-0.06889, -0.01509] | 0.01155 |
| two_concepts | TwoLayerCNN | release_default | -0.04156 | [-0.06983, -0.01329] | 0.0144 |
| two_concepts | TwoLayerCNN | release_late | -0.04523 | [-0.06854, -0.02192] | 0.001467 |

## Integrity and interpretation

Maximum full-patch numerical error: 0.
Maximum no-op numerical error: 0.

The complete selected-fidelity curves for k=1..16 are in `full_budget_curves.csv` and `selected_fidelity_budget_curves.png`. Energy-matched diagnostics remain restricted to k={1,2,4,8} as frozen in the protocol.

Release timing is interpreted as a sensitivity map, not as evidence for a monotone causal dose law. This renderer study does not establish human interpretability or natural-image generalization.
