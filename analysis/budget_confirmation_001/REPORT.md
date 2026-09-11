# Prospective intervention-budget confirmation

Protocol: `studies/cnn_budget_confirmation/PROTOCOL.md` was committed before blocks 4000–4019 were generated.

- protocol SHA-256: `94ba401db09f8a0a0102fde5a214640d7ba66c40b03916f1f5b42927f745baea`
- protocol commit: `2b142fb3a9c9883fc12b7606512de7bb2ec44a35`
- analysis source SHA-256: `8ad422e87942ad54ad6582fcb05af4cd14bcd68958b812fc8472a060432cea21`

Planned final checkpoint evaluations: **80**; loaded: **80**.

## Primary test

The single primary test is the pre-specified TinyCNN / contrast-ranking selected-fidelity budget contrast

\[B=\tfrac12[\Delta F(4)+\Delta F(8)]-\tfrac12[\Delta F(1)+\Delta F(2)].\]

- n = 20 fresh blocks
- mean B = +0.330976
- SD = 0.122424
- 95% t interval = [+0.273680, +0.388273]
- t = 12.0905
- two-sided p = 2.2819196e-10
- predeclared primary decision = **CONFIRMED**

The decision above is computed mechanically from the frozen rule: positive mean B, lower 95% interval bound above zero, and p < 0.05.

## Full selected-fidelity curve

| Architecture | Ranking | k | Δ selected F | 95% interval |
|---|---|---:|---:|---:|
| TinyCNN | contrast | 1 | -0.3459 | [-0.4033, -0.2885] |
| TinyCNN | contrast | 2 | -0.3317 | [-0.4082, -0.2552] |
| TinyCNN | contrast | 4 | -0.0221 | [-0.0424, -0.0019] |
| TinyCNN | contrast | 8 | +0.0065 | [+0.0037, +0.0093] |
| TinyCNN | auroc | 1 | -0.3040 | [-0.3662, -0.2418] |
| TinyCNN | auroc | 2 | -0.3237 | [-0.3988, -0.2485] |
| TinyCNN | auroc | 4 | -0.0332 | [-0.0696, +0.0032] |
| TinyCNN | auroc | 8 | +0.0058 | [+0.0026, +0.0091] |
| TinyCNN | validation_patch | 1 | -0.2856 | [-0.3191, -0.2520] |
| TinyCNN | validation_patch | 2 | -0.1733 | [-0.2068, -0.1399] |
| TinyCNN | validation_patch | 4 | -0.0052 | [-0.0202, +0.0098] |
| TinyCNN | validation_patch | 8 | +0.0073 | [+0.0047, +0.0098] |
| TwoLayerCNN | contrast | 1 | -0.0117 | [-0.0202, -0.0032] |
| TwoLayerCNN | contrast | 2 | -0.0378 | [-0.0493, -0.0263] |
| TwoLayerCNN | contrast | 4 | -0.1313 | [-0.1645, -0.0980] |
| TwoLayerCNN | contrast | 8 | -0.0020 | [-0.0493, +0.0453] |
| TwoLayerCNN | auroc | 1 | -0.0141 | [-0.0224, -0.0058] |
| TwoLayerCNN | auroc | 2 | -0.0438 | [-0.0565, -0.0311] |
| TwoLayerCNN | auroc | 4 | -0.1319 | [-0.1614, -0.1025] |
| TwoLayerCNN | auroc | 8 | +0.0104 | [-0.0402, +0.0611] |
| TwoLayerCNN | validation_patch | 1 | -0.0256 | [-0.0380, -0.0131] |
| TwoLayerCNN | validation_patch | 2 | -0.0760 | [-0.1048, -0.0473] |
| TwoLayerCNN | validation_patch | 4 | -0.0883 | [-0.1401, -0.0365] |
| TwoLayerCNN | validation_patch | 8 | +0.0100 | [-0.0015, +0.0214] |

## Secondary checks

Energy-matched outputs were complete. Mean relative matching error=0.0949916; maximum=3.8549.

All AUROC/validation-patch rankings, TwoLayerCNN results, absolute counterfactual accuracy, U, and energy-matched controls are secondary. They do not change the primary decision.

## Integrity

Maximum full-patch error: 0.
Maximum no-op error: 0.

This confirmation is independent in renderer blocks, not in task family or codebase. It does not establish a unique mechanism, human interpretability, or natural-image generalization.
