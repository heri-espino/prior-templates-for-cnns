# Saved findings from the completed 400-run study

**Training is complete and saved.** The [full paper-structured report](REPORT.md) and final audit are complete. There are 400 main runs: two tasks, two architectures, ten independent blocks and ten conditions. Every run has a saved model, epoch history, scalar metrics and intervention results.

## Main result

Stronger template retention increased kernel alignment in all four settings. **None of the four prospective tests established an improvement in causal usefulness after Holm correction.** This is an inconclusive causal-effect result, not proof that retention has no effect.

Treatment: `template_retention_1` versus `template_init`. Causal usefulness U is the fidelity advantage of four validation-selected concept channels over four random channels. Intervals are marginal 95% paired t intervals over ten independent blocks.

| Task / model | Alignment change | Accuracy change | Change in U [95% CI] | Holm p |
|---|---:|---:|---:|---:|
| single_shape / TinyCNN | +0.268 | −0.23 pp | +0.0345 [0.0018, 0.0672] | 0.1633 |
| single_shape / TwoLayerCNN | +0.172 | −0.04 pp | −0.0092 [−0.0513, 0.0329] | 1.0000 |
| two_concepts / TinyCNN | +0.289 | −6.13 pp | +0.0252 [−0.0306, 0.0811] | 0.9995 |
| two_concepts / TwoLayerCNN | +0.198 | −0.39 pp | −0.0074 [−0.0557, 0.0408] | 1.0000 |

The first unadjusted interval excludes zero, but its test does not survive the predeclared correction across four comparisons. The compositional shallow model also incurs a meaningful accuracy loss.

## A clearer mechanism: classifier optimization

Keeping frozen-template convolutional features unchanged and refitting only the linear classifier improves mean test accuracy:

- **single_shape / TinyCNN:** 83.40% → 99.77%.
- **two_concepts / TinyCNN:** 52.30% → 81.80%.

These are validation-selected classifier diagnostics, separate from the primary causal tests. They demonstrate that the original trained classifier did not exhaust the predictive information available in those fixed features. They do not prove a global representation ceiling for the remaining errors.

## What is preserved

- [Full per-run table](analysis/per_run.csv)
- [Four primary comparisons](analysis/primary_contrasts.csv)
- [Classifier diagnostics](analysis/head_diagnostics.csv)
- [All patching summaries](analysis/patching_all.json)
- [Frozen prospective protocol](PROTOCOL.md)
- [Experiment design and source hashes](design_freeze.json)
- `runs/`: all 400 checkpoints and their raw outputs.
- `data/`: rendered images, concept labels, masks, matched nuisance variants and context metadata.
- `worker0.log`, `worker1.log`, `execution_worker*.jsonl`: execution records.

The original first-pass project remains unchanged. The final audit passed for all 400 checkpoints and 60 dataset splits (20,480 unique images). The background-equality check was corrected to include exact antialiased rendering support; no training data or model was changed. See [the report](REPORT.md) and [audit](audit.json).
