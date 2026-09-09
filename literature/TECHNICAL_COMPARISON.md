# Technical reading of the three additions

Source snapshot: user commit `566123daaab7b85e2be36ef9460886ebf351cae9`, fetched after `36fc61b`. The new commit supplies three full Docling texts and their extracted figures/tables. The referenced new PDF paths are absent in this checkout; extraction quality labels are not independent verification. Miller and Nicolson were also read in author-hosted arXiv PDFs carrying publication headers during the preceding review. Sharma's supplied full extraction now resolves the earlier abstract-only access limitation. No source extraction or experimental artifact was modified.

## Miller, Chughtai and Saunders — COLM 2024

[Supplied text](extracted/miller_2024_transformer_faithfulness_metrics_not_robust.md). Read §§3–6, methodological appendix B and Tracr discussion E.3; primary PDF: https://arxiv.org/pdf/2407.08734.

The six choices in §3.1 separate graph granularity, component type, replacement value, token positions, direction, and circuit versus complement. Node replacement affects every downstream consumer; edge replacement isolates a destination. Branch interventions are discussed but explicitly not executed. The IOI experiments show score sensitivity to ablation methodology; the compiled Tracr examples show that the reference circuit itself depends on what information an ablation removes. Resampling can preserve information constant across the source distribution that zeroing removes.

**Implication for us:** channel count is a budget at fixed channel granularity. Our patches propagate through the complete downstream model and do not test an isolated circuit's sufficiency by corrupting its complement. U and recovered logit-difference faithfulness are different estimands. The paper is direct prior art for protocol sensitivity, not evidence that our measured reversal generalizes.

## Nicolson, Schut, Noble and Gal — TMLR 2025

[Supplied text](extracted/nicolson_2025_explaining_explainability.md). Read §§3–8 and implementation §10; primary PDF: https://arxiv.org/pdf/2404.03713.

CAVs are learned linear directions; TCAV uses directional derivatives. The authors examine layer consistency, entanglement and spatial dependence. Elements changes shape–colour associations while controlling the visual generative process. In §6.2, association between red and triangle yields elevated red TCAV scores for striped triangles despite correct classifications. Spatial probe construction changes CAV spatial norms and sensitivities (§6.3); probe-set choices also matter in the medical example (§7).

**Implication for us:** a concept score depends on how the concept and probe population are defined. Renderer labels avoid evaluating bank resemblance against itself, but do not establish that a channel uniquely implements that concept. Our AUROC/IoU are neither CAVs nor TCAV. Elements is an existing controlled visual precedent; we cannot claim synthetic concept control as novel.

## Sharma and Le — TMLR 2026

[Supplied text](extracted/sharma_2026_encoding_wo_influence.md). Read §§3–6, D.2, D.4–D.5 and F. Publication record: https://openreview.net/forum?id=TQbXHsI3Lm. The public author code covers Gemma only; the paper is the source for the other models.

They compare SAE encoding contrasts with residual-stream interventions at the response token. NPE is a signed ratio of the intervention-induced expected-answer shift to the original demographic contrast; it can exceed one. It is not our probability-response fidelity or U. §4.4 distinguishes per-pair top-K selection from aggregate selection and varies K across 5, 10, 20 and 50. Small baseline effects are excluded; steering is Winsorized. §4.5 includes active-random, shuffled-magnitude, cross-condition and variance-matched controls, with widened matching windows when needed.

The reported separation concerns layer profiles in Gemma and Qwen, with partial replication in Llama: its ablation comparison does not reach significance (§5.7). The invariant is not an early encoding peak in every model. §6.3 and D.2 document scoring sensitivity in a particular survey subset. Appendix F addresses correlated demographic content in patches; it does not establish universal monosemanticity.

**Implication for us:** the general encoding–influence distinction, multiple budgets and selection sensitivity already have close precedents. Our random-channel baseline is not an activation-variance- or perturbation-magnitude-matched baseline. Initialization spectrum/Gram matching in Stage A does not provide that intervention control. Claims about semantic specificity must retain this limitation.

## Integration and next tests

The main paper now cites all three for these bounded comparisons. Its contribution remains the controlled template-retention/release training comparison and the observed budget-dependent treatment reversal in small CNNs. No new empirical results, statistical corrections or causal-identification theorem follow from adding citations.

These are proposed tests, not completed work:

1. On saved checkpoints, compare selected patches with channel sets matched on activation variance and actual patch magnitude. Report matching balance, candidate shortages and fallback rules, alongside existing random baselines.
2. Fix concept rankings on a disjoint selection split and evaluate the complete prespecified budget family. Keep per-example selection, if added, explicitly exploratory.
3. Test semantic specificity by varying concept associations and position while retaining matched renderer interventions; evaluate effects on the target and non-target concepts.

PCFNet remains unavailable in full text: the new commit contains no PCFNet text/PDF. Publisher previews support only the existing limited description. The earlier GO-CNN preprint by the same authors has a different title and is not a substitute for the final PCFNet article.

The final independent experimental submission audit remains deferred.

## Orozco-Solis et al. (2024) follow-up

Supplied ten-page publisher PDF and corresponding user text added on 9 September. Sections 4 and 6 describe freezing the other convolutional layers, monitoring pairwise Gabor-filter distances, and proposing correlation-distance-based stopping of Gabor-layer training. Their fully connected-layer explanation is explicitly speculative. This is a filter-degradation/early-stopping precedent, not evidence for our causal intervention outcomes. Main reference 24 records the comparison.
