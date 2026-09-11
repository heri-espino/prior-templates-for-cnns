# Main manuscript status

Editable submission manuscript: [`main.tex`](main.tex). Historical Markdown sources are not automatically synchronized with the LaTeX manuscript. The new fresh-data appendix is [`confirmation_robustness_results.tex`](confirmation_robustness_results.tex); earlier Stage A--C numerical appendices remain in [`supplementary_results.tex`](supplementary_results.tex), and the kernel-matching appendix remains in [`kernel_matching_results.tex`](kernel_matching_results.tex).

## Current scientific status

The manuscript has been rewritten around the final evidential hierarchy rather than the earlier exploratory `U` reversal.

- **Stage A:** 400-model alignment/control experiment. Persistent retention strongly increases template alignment, but the four locally prospective `U` tests do not survive Holm correction.
- **Stage B:** 200-model, 200-epoch retention/release experiment. Template initialization does not uniformly accelerate learning; release reduces alignment and improves compositional accuracy relative to constant retention.
- **Stage C:** post hoc 80-checkpoint measurement analysis. Decomposition shows that the positive TinyCNN `k=4` release effect in `U_random` is substantially baseline-driven. Selected fidelity is strongly negative at small budgets, near zero/slightly negative at `k=4`, and slightly positive at `k=8`.
- **Stage D:** frozen prospective confirmation on 80 newly trained models from blocks 4000--4019. The single primary `two_concepts / TinyCNN / contrast` selected-fidelity budget statistic is **CONFIRMED**: mean `B=+0.330976`, 95% CI `[+0.273680,+0.388273]`, `p=2.2819e-10` over 20 fresh blocks.
- **Stage E:** separately frozen 1,200-model robustness map on blocks 5000--5049. The default-release budget contrast is positive for TinyCNN on both tasks (`+0.07189` single_shape; `+0.28619` two_concepts), but not for TwoLayerCNN (`-0.01018` and `-0.04156`, respectively). Full `k=1..16` curves and all six prior profiles are retained.
- **Independent checkpoint audit:** 32/32 fixed Stage-B checkpoint evaluations completed and passed all frozen tolerances after correcting an audit-only first-layer selection bug. Maximum selected-fidelity discrepancy is `2.39e-7`; full/no-op probability errors are zero.

## Current paper claim

The supported claim is narrow: **within the tested small-CNN renderer setting, the release-minus-retention treatment effect on selected first-layer intervention fidelity depends strongly on intervention budget, and that dependence is architecture-sensitive.**

The manuscript does **not** claim that release redistributes causal information, identifies a unique mechanism, yields human-interpretable filters by construction, or generalizes to natural images.

## Remaining submission preparation

The experimental confirmation and internal audit are complete. Remaining work is submission packaging rather than new outcome-driven experimentation: compile and visually inspect the revised TMLR PDF, perform coauthor/human scientific review, prepare an anonymous artifact if required, and decide whether to archive that artifact with a DOI. The current public working repository should not be linked from the anonymous manuscript because it identifies the authors.
