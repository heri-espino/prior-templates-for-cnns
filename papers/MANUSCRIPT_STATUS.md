# Main manuscript status

Editable submission manuscript: [`main.tex`](main.tex). Historical Markdown sources are not automatically synchronized with the LaTeX manuscript. Fresh-data numerical appendices are in [`confirmation_robustness_results.tex`](confirmation_robustness_results.tex); earlier Stage A--C numerical appendices remain in [`supplementary_results.tex`](supplementary_results.tex); and [`kernel_matching_results.tex`](kernel_matching_results.tex) retains the complete endpoint matching table because its visual summaries were promoted to the main Results section.

## Current title and positioning

**Template Alignment Is Not Functional Alignment: Intervention-Budget Sensitivity in Small CNNs**

The manuscript is now positioned around one conceptual question rather than around the chronology of the experiments: **does retaining an interpretable-looking first-layer feature imply a stable functional role?** The paper's answer is deliberately narrow. Template retention strongly preserves weight-space morphology, but the release-versus-retention intervention result depends on patch budget, baseline definition, and architecture.

The symbol `B` is now named the **intervention-budget contrast**. Positive `B` means release becomes relatively more faithful at the larger tested patch budgets (`k=4,8`) than at the smaller budgets (`k=1,2`); it does not imply a positive release effect at every larger `k`.

## Current scientific status

- **Stage A:** 400-model alignment/control experiment. Persistent retention strongly increases template alignment, but the four locally prospective `U` tests do not survive Holm correction.
- **Stage B:** 200-model, 200-epoch retention/release experiment. Template initialization does not uniformly accelerate learning; release reduces alignment and improves compositional accuracy relative to constant retention.
- **Stage C:** post hoc 80-checkpoint measurement analysis. Decomposition shows that the positive TinyCNN `k=4` release effect in `U_random` is substantially baseline-driven. Selected fidelity is strongly negative at small budgets, near zero/slightly negative at `k=4`, and slightly positive at `k=8`.
- **Stage D:** frozen prospective confirmation on 80 newly trained models from blocks 4000--4019. The single primary `two_concepts / TinyCNN / contrast` intervention-budget contrast is **CONFIRMED**: mean `B=+0.330976`, 95% CI `[+0.273680,+0.388273]`, `p=2.2819e-10` over 20 fresh blocks.
- **Stage E:** separately frozen 1,200-model robustness map on blocks 5000--5049. The default-release intervention-budget contrast is positive for TinyCNN on both tasks (`+0.07189` single_shape; `+0.28619` two_concepts), but not for TwoLayerCNN (`-0.01018` and `-0.04156`, respectively). Full `k=1..16` curves and all six prior profiles are retained.
- **Independent checkpoint audit:** 32/32 fixed Stage-B checkpoint evaluations completed and passed all frozen tolerances after correcting an audit-only first-layer selection bug. Maximum selected-fidelity discrepancy is `2.39e-7`; full/no-op probability errors are zero.

## Claim-first framing pass

The title, abstract, Introduction, Results headings, Interpretation, and Conclusion now foreground the central intellectual object rather than the experimental sequence.

- The abstract opens with the appearance-versus-function gap, then states the prospective intervention-budget result before summarizing the evidence hierarchy.
- The Introduction asks one central question and presents the Stage-D/Stage-E result before explaining how the earlier baseline decomposition motivated the confirmatory estimand.
- Results subsection titles are claim-driven: strong alignment is separated from functional advantage; release is framed as an alignment/accuracy tradeoff; the fixed-budget relative score is identified as baseline-sensitive; Stage D is the prospective budget-shift result; and Stage E is the architecture boundary.
- `B` is explicitly interpreted in words wherever it is introduced, so it no longer reads as an unexplained custom statistic.
- The Conclusion now ends on the operational lesson: what a feature looks like and what an intervention says it does should be reported separately.

The manuscript still does **not** claim that release redistributes causal information, identifies a unique mechanism, yields human-interpretable filters by construction, or generalizes to natural images.

## Results-layout pass

The final evidence is visible in the main argument rather than being carried by the appendices.

- The experimental-roadmap table was reduced to three columns and fits within the TMLR text width.
- The four-row Stage-E default-release architecture comparison is a main-text table, so the TinyCNN/TwoLayerCNN boundary is visible beside the robustness discussion.
- The complete Stage-E selected-fidelity budget curves remain in the main Results section and have explicit reading guidance.
- The one-to-one matching trajectories and the complete fixed-block kernel/template similarity matrices were moved from the appendix into the main Results section and are explained directly in the text.
- The matched first-layer kernel gallery remains beside those figures, giving three complementary morphology views: trajectory, full similarity matrix, and filters themselves.
- The kernel-matching appendix contains only the complete numerical endpoint table; the fresh-data appendix no longer repeats the four-row Stage-E summary already shown in the paper.

## Build and visual validation

GitHub Actions `paper-compile` run `34728686761` completed successfully at commit `3ecdea067e27fe92401f489b56125644ec1bb7c8`. The claim-first manuscript and standardized appendix terminology compile successfully under the TMLR build.

The rendered first pages were visually inspected after the positioning rewrite. The new title fits the TMLR title block, the abstract remains on the first page, and the Introduction immediately states the central question and headline result without clipping or overlap.

## Remaining submission preparation

The experimental confirmation, internal audit, compile check, layout inspection, and positioning pass are complete. Remaining work is submission packaging rather than new outcome-driven experimentation: perform coauthor/human scientific review, prepare an anonymous artifact if required, and decide whether to archive that artifact with a DOI. The current public working repository should not be linked from the anonymous manuscript because it identifies the authors.
