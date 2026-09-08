# Verified results

Study 02 primary contrast: template_retention_1 minus template_init; paired n=10 independent blocks. U = selected-k4 fidelity minus mean random-k4 fidelity. Intervals below are marginal 95% t intervals; p-values are Holm corrected across four tests.

| Setting | Δalignment | ΔU [CI] | Holm p | Δaccuracy pp |
|---|---:|---|---:|---:|
| single_shape / TinyCNN | .268 | .0345 [.0018,.0672] | .1633 | −.23 |
| single_shape / TwoLayerCNN | .172 | −.0092 [−.0513,.0329] | 1 | −.04 |
| two_concepts / TinyCNN | .289 | .0252 [−.0306,.0811] | .9995 | −6.13 |
| two_concepts / TwoLayerCNN | .198 | −.0074 [−.0557,.0408] | 1 | −.39 |

All four decisions inconclusive. Do not say all marginal intervals include zero; the first excludes zero but fails correction. Nonsignificance is not equivalence.

Frozen-template TinyCNN head refit: single_shape 83.40→99.77%; two_concepts 52.30→81.80%. Features unchanged. 398/400 selected refits report successful termination. Additional 200 head-patching evaluations were exploratory, designed after seeing partial outcomes. They do not replace primary results.

Learning curves: 400 × 40 =16,000 epoch records. Early mean validation accuracy epochs1–10 for single_shape/TwoLayerCNN: random80.57%, random_unitnorm85.91%, template_init87.88%. In two_concepts/TinyCNN templates start slower. Loss still decreases late; no established plateau. Initial/final kernels saved in study02, but no per-epoch model checkpoints or causal trajectories.

Audit: 400 checkpoints, 60 splits, 20,480 unique images; source/protocol unchanged. Background check initially missed faint antialiasing support; exact-render support corrected the audit without changing data/training.

Authoritative evidence: `studies/cnn_causal_milestone/REPORT.md`, `audit.json`, `analysis/primary_contrasts.csv`, `LEARNING_DYNAMICS.md`.

## Study03 (completed, exploratory)

200 runs, 40,000 epochs, 1,800 checkpoint evaluations. Template_init − random_unitnorm early mean accuracy: single_shape/TwoLayerCNN +3.34pp [1.25,5.42]; two_concepts/TinyCNN −6.84pp [−10.26,−3.43]. Other intervals include zero.

At epoch200 on two_concepts/TinyCNN, template_init and random_unitnorm both99.30% test accuracy. Constant retention98.01%; release99.22% (paired gain1.21pp [.71,1.71]); alignment .947→.580; U .657→.718 (Δ .061 [.027,.095]). Deeper two_concepts: accuracy99.14→99.53%; U .061→.003 (Δ−.058 [−.094,−.021]). All intervals marginal/exploratory; no corrected confirmatory inference.

Relative U and counterfactual correctness differ: shallow compositional selected patch accuracy constant93.61% vs release90.64%, despite increased U. Do not summarize as uniformly improved causal interventions. All200 runs attained95% validation at least once. All1800 selected head fits terminated successfully. Artifact validation is not full forward recomputation.
