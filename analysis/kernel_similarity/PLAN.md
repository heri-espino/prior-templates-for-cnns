# Checkpoint kernel analysis plan

Written before this analysis was executed, 9 September 2026; exploratory follow-up after the existing outcomes were known. This is not a prospective confirmatory protocol.

Scope: Stage B retention_release_001, all 200 models (two tasks, two architectures, ten blocks, five conditions), at epoch 0 and every saved scheduled checkpoint through 200. No new training or forward evaluations. Other stages and their spectrum/frozen conditions are outside this follow-up.

Use the exact corrected bank from the preserved release-experiment core. Center each kernel and normalize its L2 norm. Compare every learned kernel against every original template (16 x 16 signed cosine matrix). Keep polarity: an inverted filter is not equivalent under ReLU. Reject zero-norm or nonfinite kernels rather than assign arbitrary similarities.

Report (1) existing mean row maximum; (2) maximum-weight one-to-one assignment using the Hungarian algorithm; (3) their gap; (4) unique nearest-template count; (5) normalized Euclidean distances sqrt(2-2*cosine); (6) maximum off-diagonal inter-kernel cosine averaged over kernels; and (7) raw kernel norms. Also keep same-index drift to initialization and compare epoch 0. The bank has redundancies/rank 10: assignment and coverage measure matching to this bank, not universal semantic diversity. One-to-one matching is a permutation of all 16 templates, not a choice of 16! separately trained models.

No human-interpretability score is introduced. Join existing accuracy, concept AUROC, localization IoU, selected/random fidelity, U and counterfactual accuracy at the same checkpoint. Spearman associations are descriptive within task x architecture x condition at epoch 200 (n=10 blocks); no p-values, causal regression or pooling epochs as independent samples. Undefined correlations stay missing.

All blocks contribute to quantitative summaries (mean and sample SD). Show every condition. Gallery/matrix examples use the lowest planned block (2000), independent of outcomes, at initialization and epoch 200. Standardize the colour scale across the gallery. Comparisons are not selected by appearance or significance.

Audit that recalculated row-max alignment matches saved evaluation metrics. Hash source, design and input files; fail on missing scheduled inputs, mismatched run identity or shape, and never overwrite experimental data. Use torch.load(weights_only=True) exclusively with no unsafe fallback. This analysis does not complete the independent submission audit.
