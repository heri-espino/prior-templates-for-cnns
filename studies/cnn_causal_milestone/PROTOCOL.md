# Milestone 2: alignment versus causal usefulness

Design fixed before experimental outcome inspection. This is a local, timestamped protocol, not an external preregistration. First-pass files and results remain unchanged.

## Questions and operational definitions

Does forcing first-layer kernels to retain corrected-template alignment increase independently measured concept selectivity and the causal usefulness of a small set of concept-selected channels? Does a frozen model fail because of the trained classifier or its representation?

Alignment is maximum signed centered cosine to the corrected reference bank, averaged over kernels. Semantic selectivity is held-out unit AUROC for renderer-defined concepts, with channel and sign selected on validation. Causal usefulness is fidelity to a **matched input intervention**, beyond a same-size random-channel intervention. These operational measures are not claims of general causal identifiability.

## Foundation repairs

- New independent SeedSequence namespaces for every block, task, split, nuisance draw, model, minibatch permutation, control bank, and random intervention. All treatments within a task/architecture/block share data, head initialization, and permutations. No split or block reuses a stream.
- Every accuracy is correct-count / example-count. Balanced factorial concept states per independently sampled nuisance context.
- Correct `corner` to an actual two-ray junction, not a sum of Gaussian derivatives. Preserve `edge_*`, `corner_*`, and `ring_*` terminology. Audit zero mean, norm, Gram matrix, singular values, and corner/edge redundancy. Do not require or imply a full-rank bank.
- A common random Fourier phase transformation of the entire bank preserves per-kernel Fourier magnitude, norms, pairwise Gram matrix, and rank, while altering spatial structure. Call this `spectrum` control, not an unstructured i.i.d. random bank.

## Experiment scope

Two synthetic tasks, 32×32 images, 16 first-layer channels, 9×9 kernels:
1. `single_shape`: original labels line/circle/triangle/square; all four identities rendered under identical nuisance factors within a context.
2. `two_concepts`: one object is square/circle and the other line/triangle; label is circle_present + 2*triangle_present. Locations are swapped randomly across contexts. Toggle one object identity while holding the other identity, positions, scales, stroke widths, noise, and occlusion fixed.

Architectures: `TinyCNN` (original convolution–ReLU–global-max–linear topology) and `TwoLayerCNN` (additional 16→16 3×3 convolution and ReLU before global-max–linear). Intervene at first-layer activation maps in both. `frozen_*` freezes the first convolution only; TwoLayerCNN's second convolution remains trainable.

Ten independent blocks (0–9). Per task/block, 128 training contexts × four states = 512 images; 64 validation contexts = 256 images; 64 held-out test contexts = 256 images. Additional matched nuisance changes preserve labels. All template treatments use the corrected bank. This is a new configuration, not a rerun of the original 64×64 experiment.

Conditions: `random`, `random_unitnorm`, `frozen_random_unitnorm`, `template_init`, `template_retention_0.1`, `template_retention_1`, `frozen_templates`, `spectrum_init`, `spectrum_retention_1`, `frozen_spectrum`. Total 400 runs.

Adam, lr 0.003, batch 128, 40 epochs, final checkpoint; no test-based tuning. Retention objective is CE + λ * sum((W-W_anchor)^2)/sum(W_anchor^2), λ as in condition name. All anchors have 16 unit-norm kernels, so denominator is 16. Only first-convolution weights are penalized. Smoke runs use a separate block namespace and validate execution; no settings are changed based on their predictive outcomes.

## Independent concept measurement

Concept annotations come only from the renderer: four identity indicators for single_shape and two binary object identities for two_concepts. On validation, rank channels by absolute standardized difference in global-max first-layer activation between concept-positive and concept-negative images. Select direction from the same validation contrast. Report held-out AUROC of the best validation-selected channel per concept, averaged across concepts, and foreground-mask localization IoU using a separate channel selected by validation IoU and per-channel 95th-percentile validation activation thresholds. Localization and identity selectivity are distinct measures.

## Matched causal intervention

For every held-out nuisance context, single_shape compares all 12 directed class changes; two_concepts compares all eight directed one-bit changes. Copy selected first-layer feature-map channels from the counterfactual image into the base image and propagate through the unchanged remainder of the network.

Let p0, p1, pS denote output probabilities for base input, actual counterfactual input, and internal patch. Define pooled effect fidelity F = 1 - sum(||(pS-p0)-(p1-p0)||²) / sum(||p1-p0||²). No-op F=0 and full patch F=1 when the input effect is nonzero; negative F is possible. If the denominator is <1e-10, mark fidelity undefined and report this condition. This measures reproduction of the model's input effect, not correctness of that effect.

Primary k=4/16 channels; compare validation concept-selected channels with eight fixed random channel rankings shared across conditions. Primary causal usefulness U = F_selected - mean(F_random). Secondary k=1 and k=8, counterfactual-label accuracy, agreement with the actual counterfactual prediction, probability-effect size, and full/no-op controls. Report all-pair outcomes; conditionally correct-pair diagnostics are supplementary and have model-dependent denominators.

Matched nuisance interventions provide a specificity control: change translation/noise while preserving identities and occluder placement and evaluate label stability and feature response. Nuisance controls are not treated as negative causal effects by assumption; the measured model may be nuisance-sensitive.

## Primary estimand and decision rule

Within each of four task×architecture settings, estimate the paired block-level difference in U between template_retention_1 and template_init. Report mean difference, 95% t interval across ten independent blocks, and two-sided paired t p-values with Holm adjustment over these four primary contrasts. All other contrasts and correlations are exploratory. Report accuracy and alignment beside U; do not pool architectures/tasks as independent seeds.

A useful manipulation requires a positive alignment change; an increase >=0.10 is a descriptive strong-manipulation threshold. Large accuracy loss (>5 percentage points) is flagged as an accuracy/retention trade-off, not suppressed. Positive adjusted evidence for U with retained predictive competence supports the hypothesis within that setting. A verified alignment increase with a negative U effect contradicts a monotonic alignment-benefit claim in that setting. Otherwise the result is inconclusive. Null results are not proof of no effect or equivalence.

## Optimization versus representation diagnostic

After training, extract the final pooled representation, standardize it using training statistics, and refit a multinomial linear classifier with L-BFGS at L2 strengths 0, 0.001, and 0.1. Choose only by validation cross-entropy. Report convergence status, gradient norm, train/validation/test accuracy and loss. Keep these diagnostic predictions separate from main causal scores. For TinyCNN with frozen first layer this directly tests classifier optimization at fixed features; for TwoLayerCNN it holds the learned second layer fixed and does not optimize the full remaining network globally. Low refit accuracy alone does not prove an information-theoretic capacity limit.

## Required checks and artifacts

Retain source/config/environment hashes, split and image hashes, masks and nuisance metadata, checkpoints, per-epoch logs, all per-block scalar results, paired effects, intervention arrays, diagnostics, figures, paper-structured Markdown report, and runnable code. Verify disjoint streams/images, exact pairing, label balance, sample weighting, corner geometry, spectrum/Gram preservation, frozen invariance, retention gradients, no-op/full patch identities, and analytic fidelity identities. Report failures and limitations without inventing results.
