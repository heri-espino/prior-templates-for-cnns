# TMLR adaptation — updated 10 September 2026

The manuscript uses the supplied official TMLR anonymous review style and bibliography files without modifying them. The class remains `article`, 10pt, with author–year citations and no identifying author block or project URL.

## Scientific revision now integrated

The TMLR manuscript has been rewritten around the final evidential hierarchy rather than the earlier exploratory four-channel `U` result.

- Stage A retains the 400-model alignment/control experiment and its four locally prospective Holm-adjusted tests.
- Stage B retains the 200-model longitudinal initialization/retention/release study.
- Stage C is explicitly post hoc. Its decomposition now shows selected fidelity and random-baseline fidelity separately and states that the positive TinyCNN `k=4` release effect in `U_random` is substantially baseline-driven.
- Stage D is the fresh frozen prospective confirmation. All 80 planned models/evaluations completed, and the single primary `two_concepts / TinyCNN / contrast` selected-fidelity budget statistic is confirmed: `B=+0.330976`, 95% CI `[+0.273680,+0.388273]`, `p=2.2819e-10`, `n=20` fresh blocks.
- Stage E is a separately frozen secondary robustness map with all 1,200 planned models/evaluations complete. It reproduces positive budget contrasts for TinyCNN across both tasks and shows that the direction is architecture-dependent rather than universal.
- The independent 32-checkpoint audit is complete and passes every frozen tolerance after correction of an audit-only first-layer selection implementation error.

The manuscript does not claim that release redistributes causal information, identifies a unique mechanism, produces human interpretability, or generalizes to natural images.

## Appendix organization

- `confirmation_robustness_results.tex`: complete fresh Stage-D results, Stage-E profile robustness summary, and checkpoint-audit tolerances/results.
- `supplementary_results.tex`: compact historical Stage A–C results explaining why the primary estimand was narrowed to selected fidelity.
- `kernel_matching_results.tex`: exploratory one-to-one template-matching follow-up from saved Stage-B checkpoints.

## Build validation

GitHub Actions workflow `paper-compile` compiled the revised anonymous TMLR manuscript successfully with pdfLaTeX/BibTeX on 10 September 2026. The final pass produced a 20-page PDF and resolved the citations and cross-references. The compiler reports one small overfull box (about 10.4 pt) in the compact experimental-stage summary table plus ordinary underfull-box/float-placement warnings; these are layout issues for final visual copyediting, not scientific or reference-resolution failures.

## Remaining author actions before submission

Perform the final visual/copyediting pass, obtain coauthor approval, and prepare an anonymous supporting artifact if required. The current public working repository should not be linked from the anonymous paper because it identifies the authors. Submission-form items such as authorship confirmation, OpenReview profiles, conflicts, funding, and required declarations remain author actions and are not invented in the manuscript.
