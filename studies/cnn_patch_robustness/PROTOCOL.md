# Patching robustness follow-up

This plan is written before running the new selection analyses, with knowledge of study03 results. It is a post hoc robustness study, not an independent confirmatory replication.

Default inputs: epoch200 from template_retention_1 and template_release, both tasks, both architectures, blocks2000–2009 (80 checkpoints). No retraining or new data. Do not pool these outcomes with study03 as independent evidence.

Select channels on validation data using (1) original standardized contrast, (2) absolute AUROC deviation from0.5, (3) individual-channel matched-patch fidelity separately within each destination identity/toggled concept. The third method ranks singleton effects; it is not greedy optimization of joint subsets. Test all rankings at k1,2,4,8; compare against eight same-size random sets generated with the original random seed namespace. Selection never uses test outcomes. If a validation concept patch denominator is undefined, mark the intervention-selected U undefined. Full/no-op checks use actual model forwards.

Report selected and random fidelity, their difference U, absolute counterfactual accuracy, agreement, correct-pair conditional accuracy, and test accuracy. Preserve rankings, singleton scores and random outcomes. Main interest: whether the release-versus-constant architecture-dependent direction persists across selection methods and sizes. Do not pick the method/size that best supports a preferred conclusion. Exploratory paired block intervals are marginal, not multiplicity-adjusted.

GPU implementation uses float32 without mixed precision or TF32; bounded batches reduce memory. Load one checkpoint at a time, retaining validation/test first-layer maps on device. CPU and GPU floating-point differences are possible; log differences against prior contrast/k4 CPU outcomes. Resume after each completed checkpoint, guarded by source/protocol/input hashes and device/configuration. A failed checkpoint is reevaluated; no partial result is marked complete.

Only after inspecting robustness should a distinct annotated task and fresh-block confirmatory experiment be finalized. This code does not claim to implement that conditional second stage.
