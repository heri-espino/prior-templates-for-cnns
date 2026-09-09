# Independent checkpoint audit

Planned and completed checkpoint evaluations: **32**.

This audit independently reimplemented the central metric computations and read stored scalar results only after recomputation.

Overall status: **PASS**.

## Maximum absolute discrepancies

| Metric | Maximum absolute difference | Tolerance |
|---|---:|---:|
| `id_acc` | 0 | 1e-12 |
| `alignment` | 2.22044604925e-16 | 1e-06 |
| `semantic_auc` | 0 | 1e-09 |
| `localization_iou` | 0 | 1e-09 |
| `fidelity_selected` | 2.38794317986e-07 | 2e-06 |
| `fidelity_random` | 1.13983348954e-07 | 2e-06 |
| `causal_usefulness` | 2.18894791487e-07 | 2e-06 |
| `cf_accuracy` | 0 | 1e-12 |
| `cf_accuracy_random` | 0 | 1e-12 |

## Identity checks

Maximum full-patch probability error: 0.
Maximum no-op probability error: 0.

## Audit conclusion

All sampled central metrics and patch identities are within the tolerances fixed in `studies/cnn_checkpoint_audit/AUDIT_PROTOCOL.md`. This supports describing the sampled Stage B checkpoint evidence as independently recomputed. It does not validate unsampled checkpoints or broader scientific claims.

## Audit-history note

The first audit execution failed only for TwoLayerCNN quantities that depend on validation-selected first-layer channels. Inspection showed that the independent audit implementation had mistakenly ranked the post-second-layer pooled representation in TwoLayerCNN, whereas the frozen audit protocol and manuscript define those concept selections on first-layer global-max activations. Accuracy, alignment, localization, random-patch fidelity, and full/no-op identities already matched in that failed diagnostic run. The implementation was corrected without altering the frozen protocol; the subsequent 32-checkpoint run above passed. Both workflow runs remain in GitHub Actions history.