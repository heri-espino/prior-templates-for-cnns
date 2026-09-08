# Does template alignment imply causal usefulness? A controlled second milestone

**Status:** completed experimental milestone; scope-limited scientific evidence, not a claim of causal identifiability or established novelty.  
**Design:** two synthetic tasks × two architectures × ten independent blocks × ten conditions.  
**Provenance:** continuation of the supplied CNN project; the original first-pass experiment is preserved in `../cnn_first_pass/`.

## Abstract

We test whether maintaining template-like convolutional kernels improves independently measured concepts and causal usefulness. A corrected template bank, independent dataset streams and sample-weighted metrics support 400 runs across two synthetic tasks, two architectures, ten independent blocks and ten conditions. Renderer annotations define concepts, and matched image interventions provide targets for activation patching. Strong retention increases alignment by 0.172–0.289 relative to template initialization across all four settings. None of the four primary causal-usefulness contrasts is significant after Holm correction; the estimates remain compatible with modest benefits or harms. In the compositional TinyCNN setting, retention also reduces accuracy by 6.13 percentage points. A fixed-feature classifier diagnostic raises frozen-template TinyCNN accuracy from 83.40% to 99.77% on single_shape and from 52.30% to 81.80% on two_concepts. These findings separate kernel appearance, concept measurements and readout optimization. They do not establish that alignment is causally useless, or that the observed dissociation is novel or general.


## 1. Introduction

The first pass found that `template_init` retained more kernel alignment than `random` at similarly high observed in-distribution accuracy, while its head-level intervention metrics did not consistently improve. It also revealed duplicated corner filters, dependent adjacent dataset seeds, unequal batch weighting, and a patching metric that counted positive margin movement as success. These issues motivated a new experiment rather than a stronger claim based on the same measurements.

The present question is operational: **when a prior increases kernel alignment, does a small set of independently selected concept channels become more useful for reproducing a matched input intervention?** We distinguish (i) resemblance to a template bank, (ii) selectivity for a renderer-defined concept, and (iii) causal usefulness under a specified intervention and output metric. The study tests whether these move together; it does not assume that one implies the others.

Structured initialization and retained filter structure have prior art, including [Molaei and Shiri Ahmad Abadi](https://doi.org/10.1016/j.asoc.2019.105960), while [Linse et al.](https://arxiv.org/abs/2411.18388) study networks with fixed predefined spatial filters. The current contribution is an experimental comparison, not a claim to have invented structured filters. Concept interventions have a conceptual precedent in [CaCE](https://arxiv.org/abs/1907.07165). Interpretation of activation patching depends on the intervention and metric, as emphasized by [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255). Our fidelity statistic below is a study-defined normalized reconstruction score; it is not presented as the CaCE estimator.

## 2. Protocol and foundation repairs

`PROTOCOL.md` and the training/evaluation source were hashed before any main run. Their hashes and the local timestamp are stored in `design_freeze.json`. This is a local prospective protocol, not an externally registered study. Smoke runs used block 100000 and two epochs solely for execution checks and timing; they are excluded. The final source and protocol hashes are checked again after training.

All results use correct-count divided by example-count. Dataset, model, shuffle, nuisance and control generators have separate named seed namespaces. Blocks 0–9 are independent dataset/model/control blocks, while treatments within each task/architecture/block share images, head initialization and minibatch permutations. Model namespaces differ between architectures, so architectural comparisons are not identical-weight interventions. The same block's tasks and architectures are not pooled as independent replications.

The audit checks every generated main image for exact duplicates across train, validation, test, tasks and blocks; it checks each stream identifier and every balanced label count. Within a context, states deliberately share the exogenous nuisance realization. Those matched observations are dependent and are not treated as independent statistical replicates.

### 2.1 Corrected templates and matched controls

The bank retains 16 entries named `edge_00`…`edge_07`, `corner_00`…`corner_03`, and `ring_00`…`ring_03`, each 9×9, zero-mean and unit L2 norm. Edges retain the original Gaussian-derivative form; rings retain the original radial difference construction. Corners are now Gaussian-weighted two-ray junctions at four right-angle orientations. They are no longer sums of orthogonal Gaussian derivatives, which had collapsed into other edges in the source archive.

The corrected bank has numerical rank 10 at tolerance 10⁻⁶. This is audited rather than hidden: eight oriented derivative edges still span two linear directions. No full-rank or complete geometric basis claim is made. The bank's concepts are hand-designed local structures; the renderer's object identities are defined separately.

For the `spectrum` control, all kernels receive the same random, conjugate-symmetric Fourier phase multiplier. This transformation preserves each kernel's Fourier magnitude and L2 norm, the full inter-kernel Gram matrix, and the bank's singular values/rank, while changing its spatial appearance. A different phase draw is used per block. These matching properties hold at initialization and remain exact when frozen; trainable kernels may depart from their initial spectra and rank. This is a constrained scrambled bank, not an i.i.d. random bank; phase scrambling does not necessarily remove all useful geometric information.

![Corrected bank and spectrum controls](figures/bank_audit.png)

## 3. Methods

### 3.1 Tasks, concepts and matched images

All images are 32×32. This smaller configuration enables the factorial study and is explicitly different from the 64×64 first pass. Shapes are drawn at 3× resolution, then downsampled. Rotation is uniform from 0 to π/4, stroke width 0.8–1.6 pixels, scale 0.85–1.15, and Gaussian noise SD 0–0.05. A 5×5 erasing rectangle is present with probability 0.1. Exact positions, scales, angle, noise hash and occluder coordinates are retained. Renderer masks identify the visible foreground of concept-positive objects, with a predeclared raster threshold of 0.2 after downsampling.

- **`single_shape`:** labels remain `line`, `circle`, `triangle`, `square`. Each independently drawn context produces all four identities with the same translation (±2 pixels), angle, scale, stroke width, noise and occluder. The radius parameter is 10×scale. There are four identity-indicator concepts. Changing an identity changes two one-hot indicators; it is a single categorical intervention, not a one-bit intervention.
- **`two_concepts`:** one object is square or circle and the other is line or triangle, with radius 4.5×scale. Their nominal centers are (8,15) and (24,17), with shared jitter; which slot contains each object family is randomly swapped per context. The label is `circle_present + 2*triangle_present`. Each context contains all four combinations. A counterfactual toggles exactly one binary concept, preserving the other object and nuisance factors.

Each task/block has 128 training contexts (512 images), 64 validation contexts (256 images), and 64 held-out test contexts (256 images). This factorial design balances labels and separates identities from nuisances. It also makes the synthetic tasks more controlled than ordinary randomly sampled image datasets. Test images and labels are never used to rank channels, set activation thresholds, choose regularization, or select checkpoints.

Additional matched nuisance images change translation and pixel-noise realization jointly, preserving labels, angle, scale, width and occluder placement. They are a joint nuisance test, not separate causal estimates for each nuisance variable. Main claims concern matched identity interventions; the original rotation/thickness OOD suites are not reproduced in this milestone.

![Matched input interventions and renderer masks](figures/matched_inputs.png)

### 3.2 Architectures and conditions

`TinyCNN` preserves the source topology: bias-free 1→16 convolution (9×9, same padding), ReLU, global spatial maximum, and a 16→4 linear head with bias. `TwoLayerCNN` inserts a bias-free 16→16 3×3 convolution and ReLU before global maximum pooling. They contain 1,364 and 3,668 parameters respectively. Both are shallow CNNs; this is a test of added depth, not broad coverage of modern architectures.

| Condition | First-layer initialization | First-layer updates | Retention λ |
|---|---|---|---|
| `random` | default PyTorch random | trainable | 0 |
| `random_unitnorm` | same random draw, centered/unit norm | trainable | 0 |
| `frozen_random_unitnorm` | centered/unit-norm random | frozen | — |
| `template_init` | corrected template bank | trainable | 0 |
| `template_retention_0.1` | corrected template bank | trainable with anchor penalty | 0.1 |
| `template_retention_1` | corrected template bank | trainable with anchor penalty | 1 |
| `frozen_templates` | corrected template bank | frozen | — |
| `spectrum_init` | matched phase-scrambled bank | trainable | 0 |
| `spectrum_retention_1` | matched phase-scrambled bank | trainable with anchor penalty | 1 |
| `frozen_spectrum` | matched phase-scrambled bank | frozen | — |

“Frozen” always refers to the first convolution. The second convolution, when present, remains trainable. This distinction matters for the optimization diagnostic and for comparing architectures.

### 3.3 Training and retention

Each main model is trained for 40 epochs, batch size 128, Adam learning rate 0.003, no scheduler, no weight decay and no early stopping. Four batches per epoch yield 160 optimizer steps. The final checkpoint is used throughout the main analysis. Every epoch stores sample-weighted training cross-entropy, accuracy, retention penalty, validation cross-entropy and validation accuracy.

The objective is

\[
L = L_{CE} + \lambda\frac{\|W-W_{anchor}\|_F^2}{\|W_{anchor}\|_F^2}.
\]

Only the first convolution is anchored. All structured anchors contain 16 unit-norm kernels, so the denominator is 16. λ=0 leaves the initialized kernels free; frozen conditions prevent all first-layer updates. This is a new soft-retention experiment, unlike the original archive, which contained initialization and freezing only.

### 3.4 Measurements independent of kernel appearance

**Alignment.** For every first-layer kernel, mean-center and compute signed cosine similarity to every corrected template. Average the maximum over kernels. Multiple kernels may choose the same template. Alignment is measured against the corrected reference bank even for `spectrum` conditions; it is not alignment to each model's own anchor. Drift L2 is also retained.

**Concept selectivity.** Compute each channel's globally maximum-pooled first-layer activation. On validation images, rank channels by the absolute positive-versus-negative concept mean difference divided by that channel's overall SD. Choose the best channel and sign on validation, then measure its AUROC on test concepts. Average over the four categorical indicators or two binary concepts. This is single-unit identity selectivity, not disentanglement or a complete concept bottleneck.

**Localization.** Independently select the channel with highest validation foreground IoU for each concept. The activation threshold for each channel is its 95th percentile over all validation images and pixels. Apply the selected channel and threshold to concept-positive test images and calculate pooled intersection/union with visible renderer masks. Average across concepts. The localization channel need not equal the selectivity channel. Neither selection procedure uses the template bank or model-head weights.

### 3.5 Matched causal patching

For `single_shape`, evaluate all 12 directed identity changes per held-out context (768 pairs). For `two_concepts`, evaluate all eight directed one-bit changes per context (512 pairs). Pairs share the exact nuisance context. Internal patching copies selected first-layer **activation maps** from the counterfactual into the base run and propagates through the unchanged remainder of the network. In the deeper model this includes its second convolution, rather than an algebraic shortcut through the final head.

Channel ranking uses the validation concept contrast above: the destination identity for a categorical change, or the toggled binary concept for a compositional change. The primary intervention patches four of sixteen channels. Eight fixed random class/concept-specific rankings provide a same-size baseline; these draws are shared across conditions within a task/block. Selected k=1 and k=8 are secondary. Random baselines are evaluated at k=4.

Let p₀ be the base probability vector, p₁ the probability vector after the actual image intervention, and pₛ the internally patched probability vector. Define effect fidelity

\[
F=1-\frac{\sum_{pairs}\|p_S-p_1\|_2^2}{\sum_{pairs}\|p_1-p_0\|_2^2}.
\]

This is algebraically the reconstruction error between internal and input-induced probability changes. No-op F=0; full-channel patch F=1 when the denominator is nonzero. For example, moving halfway from p₀ to p₁ gives F=0.75. Thus F is an error-reduction score, not the fraction of a causal effect mediated. Negative values mean that the patch is farther from the actual counterfactual output than leaving the model unchanged. If the denominator is below 10⁻¹⁰, fidelity is undefined and cannot support a causal-usefulness estimate.

The primary causal-usefulness measure is

\[
U=F_{selected,k=4}-\frac{1}{8}\sum_{r=1}^{8}F_{random_r,k=4}.
\]

It asks whether validation-selected concept channels are more useful than a random set of the same size. It is not the total usefulness of the network or an absolute causal score for every aligned kernel. We also report counterfactual-label accuracy, agreement with the actual counterfactual prediction, correct-pair subset diagnostics, input-effect magnitude, and k dependence. Fidelity to a model's erroneous counterfactual remains possible, so predictive accuracy must accompany F and U.

### 3.6 Statistical analysis and decision rules

The four prospective primary contrasts are `template_retention_1 − template_init` in U, one per task×architecture. Each uses paired differences across ten independent blocks, a mean, a marginal 95% Student-t interval and a two-sided paired t-test. Holm adjustment controls the family of four primary p-values. Intervals displayed are marginal, not simultaneous Holm-adjusted intervals. t-based inference assumes reasonably behaved block differences; ten blocks provide limited precision and distributional diagnostics.

A positive alignment change establishes the direction of the manipulation; ≥0.10 is the predeclared descriptive threshold for a strong manipulation. A loss of more than five accuracy percentage points is flagged rather than removed. An adjusted positive U effect without that accuracy loss supports the operational hypothesis in that setting. An adjusted negative U effect despite increased alignment contradicts a monotonic benefit in that setting. Other outcomes are inconclusive. A nonsignificant result is not equivalence or evidence that no effect exists.

Other condition contrasts, localization/selectivity effects, pooled-treatment correlations, classifier-refit gains, and k curves are exploratory. Correlation plots do not treat all treatment runs as independent and do not identify an effect of alignment itself: changing a retention penalty changes other properties too.

### 3.7 Optimization versus representation diagnostic

After each main run, hold the learned representation fixed and fit an alternative multinomial linear head. Standardize pooled features using training means/SDs. Use L-BFGS for up to 500 iterations at head L2 strengths 0, 0.001, and 0.1, selecting only by validation cross-entropy. Save all candidates, termination messages, gradient norms, coefficients and selected predictions. This L2 strength is separate from convolution-retention λ.

For frozen `TinyCNN`, a successful accuracy recovery demonstrates that the original Adam-trained head did not exhaust the available representation. For `TwoLayerCNN`, this diagnostic holds its learned second convolution fixed too; it does not solve the full downstream nonconvex optimization problem. Failure of the refit to improve cannot prove that the input information is absent, especially if optimization did not converge. Main causal metrics always use the original trained head; refit results are diagnostic only. An explicitly exploratory supplement, designed after inspecting a 189-run partial table, repeats the same patches using the already validation-selected refitted head for five conditions. It changes no features, channel rankings or primary tests. Its purpose is to assess classifier and probability-calibration sensitivity; see `EXPLORATORY_SUPPLEMENT.md`.

## 4. Results

### 4.1 Completion and validity

400 runs are included. All 400 planned runs completed; every condition has ten blocks in each setting. The final audit independently verifies sample-weighted checkpoint accuracy, alignment, full/no-op patch identities, classifier-refit predictions, exact data pairing, and source/protocol integrity. The first-pass report and uploaded-source working files remain unchanged.

### 4.2 Predictive accuracy, alignment, semantics and causal usefulness

Values are mean ± sample SD across independent blocks. Accuracy is percent; alignment, AUROC, IoU and U are dimensionless. U is selected k=4 fidelity minus the eight-random-ranking mean. All labels and nomenclature match this milestone’s code.

**single_shape / TinyCNN**

| Condition | ID accuracy (%) | Alignment | Concept AUROC | Mask IoU | U |
| --- | --- | --- | --- | --- | --- |
| `random` | 100.00 ± 0.00 | 0.308 ± 0.032 | 0.976 ± 0.016 | 0.276 ± 0.072 | 0.268 ± 0.068 |
| `random_unitnorm` | 100.00 ± 0.00 | 0.293 ± 0.035 | 0.974 ± 0.017 | 0.277 ± 0.061 | 0.286 ± 0.061 |
| `frozen_random_unitnorm` | 74.34 ± 12.68 | 0.153 ± 0.016 | 0.862 ± 0.019 | 0.239 ± 0.081 | 0.259 ± 0.067 |
| `template_init` | 100.00 ± 0.00 | 0.653 ± 0.027 | 0.971 ± 0.013 | 0.427 ± 0.048 | 0.254 ± 0.061 |
| `template_retention_0.1` | 100.00 ± 0.00 | 0.761 ± 0.012 | 0.965 ± 0.013 | 0.499 ± 0.047 | 0.266 ± 0.053 |
| `template_retention_1` | 99.77 ± 0.27 | 0.921 ± 0.007 | 0.946 ± 0.011 | 0.550 ± 0.034 | 0.288 ± 0.060 |
| `frozen_templates` | 83.40 ± 7.84 | 1.000 ± 0.000 | 0.888 ± 0.012 | 0.561 ± 0.044 | 0.252 ± 0.077 |
| `spectrum_init` | 100.00 ± 0.00 | 0.353 ± 0.075 | 0.970 ± 0.011 | 0.287 ± 0.079 | 0.287 ± 0.049 |
| `spectrum_retention_1` | 99.80 ± 0.28 | 0.413 ± 0.089 | 0.946 ± 0.014 | 0.273 ± 0.124 | 0.295 ± 0.053 |
| `frozen_spectrum` | 82.93 ± 6.90 | 0.416 ± 0.092 | 0.850 ± 0.023 | 0.249 ± 0.139 | 0.296 ± 0.076 |

**single_shape / TwoLayerCNN**

| Condition | ID accuracy (%) | Alignment | Concept AUROC | Mask IoU | U |
| --- | --- | --- | --- | --- | --- |
| `random` | 100.00 ± 0.00 | 0.282 ± 0.034 | 0.851 ± 0.021 | 0.305 ± 0.048 | 0.039 ± 0.029 |
| `random_unitnorm` | 100.00 ± 0.00 | 0.245 ± 0.030 | 0.851 ± 0.019 | 0.294 ± 0.053 | 0.029 ± 0.044 |
| `frozen_random_unitnorm` | 99.92 ± 0.16 | 0.151 ± 0.028 | 0.861 ± 0.029 | 0.246 ± 0.043 | 0.007 ± 0.027 |
| `template_init` | 99.96 ± 0.12 | 0.824 ± 0.011 | 0.843 ± 0.026 | 0.480 ± 0.038 | 0.058 ± 0.040 |
| `template_retention_0.1` | 99.96 ± 0.12 | 0.950 ± 0.007 | 0.838 ± 0.024 | 0.522 ± 0.039 | 0.062 ± 0.062 |
| `template_retention_1` | 99.92 ± 0.16 | 0.996 ± 0.001 | 0.877 ± 0.016 | 0.549 ± 0.043 | 0.049 ± 0.058 |
| `frozen_templates` | 99.96 ± 0.12 | 1.000 ± 0.000 | 0.888 ± 0.012 | 0.561 ± 0.044 | 0.049 ± 0.050 |
| `spectrum_init` | 100.00 ± 0.00 | 0.371 ± 0.081 | 0.851 ± 0.016 | 0.270 ± 0.105 | 0.061 ± 0.059 |
| `spectrum_retention_1` | 99.96 ± 0.12 | 0.415 ± 0.094 | 0.845 ± 0.022 | 0.255 ± 0.135 | 0.059 ± 0.061 |
| `frozen_spectrum` | 99.96 ± 0.12 | 0.416 ± 0.092 | 0.850 ± 0.023 | 0.249 ± 0.139 | 0.061 ± 0.055 |

**two_concepts / TinyCNN**

| Condition | ID accuracy (%) | Alignment | Concept AUROC | Mask IoU | U |
| --- | --- | --- | --- | --- | --- |
| `random` | 98.20 ± 1.29 | 0.247 ± 0.030 | 0.984 ± 0.011 | 0.174 ± 0.028 | 0.578 ± 0.054 |
| `random_unitnorm` | 97.58 ± 1.93 | 0.233 ± 0.027 | 0.979 ± 0.012 | 0.180 ± 0.034 | 0.570 ± 0.044 |
| `frozen_random_unitnorm` | 45.00 ± 6.64 | 0.153 ± 0.015 | 0.741 ± 0.042 | 0.191 ± 0.029 | 0.277 ± 0.111 |
| `template_init` | 96.99 ± 2.53 | 0.657 ± 0.036 | 0.977 ± 0.008 | 0.277 ± 0.032 | 0.555 ± 0.092 |
| `template_retention_0.1` | 96.56 ± 3.43 | 0.772 ± 0.026 | 0.962 ± 0.013 | 0.302 ± 0.023 | 0.558 ± 0.099 |
| `template_retention_1` | 90.86 ± 8.16 | 0.946 ± 0.008 | 0.909 ± 0.041 | 0.355 ± 0.015 | 0.580 ± 0.099 |
| `frozen_templates` | 52.30 ± 7.57 | 1.000 ± 0.000 | 0.769 ± 0.009 | 0.399 ± 0.010 | 0.679 ± 0.121 |
| `spectrum_init` | 96.80 ± 2.04 | 0.330 ± 0.070 | 0.980 ± 0.010 | 0.187 ± 0.042 | 0.588 ± 0.069 |
| `spectrum_retention_1` | 87.11 ± 3.98 | 0.408 ± 0.095 | 0.935 ± 0.022 | 0.179 ± 0.044 | 0.600 ± 0.090 |
| `frozen_spectrum` | 47.19 ± 10.57 | 0.416 ± 0.092 | 0.725 ± 0.045 | 0.174 ± 0.062 | 0.373 ± 0.175 |

**two_concepts / TwoLayerCNN**

| Condition | ID accuracy (%) | Alignment | Concept AUROC | Mask IoU | U |
| --- | --- | --- | --- | --- | --- |
| `random` | 98.83 ± 1.09 | 0.240 ± 0.034 | 0.796 ± 0.043 | 0.222 ± 0.023 | 0.173 ± 0.067 |
| `random_unitnorm` | 99.18 ± 0.95 | 0.219 ± 0.025 | 0.773 ± 0.051 | 0.221 ± 0.027 | 0.158 ± 0.144 |
| `frozen_random_unitnorm` | 99.02 ± 0.81 | 0.154 ± 0.020 | 0.758 ± 0.027 | 0.171 ± 0.032 | 0.064 ± 0.078 |
| `template_init` | 98.79 ± 1.36 | 0.789 ± 0.014 | 0.788 ± 0.036 | 0.326 ± 0.009 | 0.095 ± 0.080 |
| `template_retention_0.1` | 98.71 ± 1.35 | 0.899 ± 0.013 | 0.777 ± 0.031 | 0.350 ± 0.017 | 0.074 ± 0.074 |
| `template_retention_1` | 98.40 ± 1.82 | 0.987 ± 0.003 | 0.768 ± 0.014 | 0.385 ± 0.009 | 0.088 ± 0.060 |
| `frozen_templates` | 96.80 ± 3.11 | 1.000 ± 0.000 | 0.769 ± 0.009 | 0.399 ± 0.010 | 0.102 ± 0.066 |
| `spectrum_init` | 99.18 ± 0.79 | 0.345 ± 0.084 | 0.778 ± 0.053 | 0.199 ± 0.053 | 0.022 ± 0.064 |
| `spectrum_retention_1` | 97.62 ± 1.55 | 0.408 ± 0.092 | 0.726 ± 0.050 | 0.192 ± 0.063 | 0.021 ± 0.157 |
| `frozen_spectrum` | 96.68 ± 1.82 | 0.416 ± 0.092 | 0.725 ± 0.045 | 0.174 ± 0.062 | 0.056 ± 0.167 |

![Alignment versus causal usefulness](figures/alignment_usefulness.png)

![Alignment versus semantic selectivity](figures/alignment_semantics.png)

### 4.3 Prospective retention contrasts

Treatment is `template_retention_1`; reference is `template_init`. Δaccuracy is in percentage points. ΔU intervals are marginal 95% intervals; p-values are Holm-adjusted across these four tests.

| Task / architecture | Δalignment | Δaccuracy pp | ΔU [95% CI] | Holm p | Operational decision |
| --- | --- | --- | --- | --- | --- |
| single_shape / TinyCNN | +0.268 | -0.23 | +0.035 [+0.002, +0.067] | 0.1633 | inconclusive causal-usefulness effect |
| single_shape / TwoLayerCNN | +0.172 | -0.04 | -0.009 [-0.051, +0.033] | 1 | inconclusive causal-usefulness effect |
| two_concepts / TinyCNN | +0.289 | -6.13 | +0.025 [-0.031, +0.081] | 0.9995 | inconclusive causal-usefulness effect |
| two_concepts / TwoLayerCNN | +0.198 | -0.39 | -0.007 [-0.056, +0.041] | 1 | inconclusive causal-usefulness effect |

Positive or negative decisions apply to the predeclared U measure and setting. A negative effect does not prove every aligned channel is useless. A positive effect does not establish a generally identified causal representation.

![Primary paired effects](figures/primary_effects.png)

### 4.4 Actual counterfactual prediction and fidelity

This table focuses on the primary treatment pair and its spectrum counterpart. “Full CF accuracy” equals model ID accuracy because incoming counterfactual states are balanced. High fidelity alone does not imply a correct counterfactual prediction.

| Setting | Condition | Selected F | Random F | Selected CF accuracy (%) | Random CF accuracy (%) | Full CF accuracy (%) | Mean squared input effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_shape / TinyCNN | template_init | 0.514 ± 0.055 | 0.260 ± 0.022 | 34.10 ± 5.60 | 6.55 ± 1.22 | 100.00 ± 0.00 | 1.725 ± 0.069 |
| single_shape / TinyCNN | template_retention_1 | 0.637 ± 0.053 | 0.349 ± 0.018 | 46.60 ± 5.35 | 10.10 ± 2.47 | 99.77 ± 0.27 | 0.832 ± 0.099 |
| single_shape / TinyCNN | spectrum_retention_1 | 0.649 ± 0.045 | 0.354 ± 0.014 | 46.71 ± 7.04 | 9.95 ± 2.49 | 99.80 ± 0.28 | 0.755 ± 0.097 |
| single_shape / TwoLayerCNN | template_init | 0.186 ± 0.049 | 0.128 ± 0.024 | 5.46 ± 5.07 | 1.17 ± 0.85 | 99.96 ± 0.12 | 1.977 ± 0.010 |
| single_shape / TwoLayerCNN | template_retention_1 | 0.276 ± 0.058 | 0.227 ± 0.025 | 9.61 ± 7.16 | 1.93 ± 1.39 | 99.92 ± 0.16 | 1.862 ± 0.032 |
| single_shape / TwoLayerCNN | spectrum_retention_1 | 0.278 ± 0.069 | 0.219 ± 0.015 | 6.24 ± 5.25 | 2.11 ± 1.16 | 99.96 ± 0.12 | 1.879 ± 0.032 |
| two_concepts / TinyCNN | template_init | 0.866 ± 0.040 | 0.311 ± 0.069 | 76.13 ± 8.36 | 13.73 ± 4.84 | 96.99 ± 2.53 | 0.993 ± 0.166 |
| two_concepts / TinyCNN | template_retention_1 | 0.911 ± 0.037 | 0.331 ± 0.079 | 76.41 ± 10.08 | 17.44 ± 4.80 | 90.86 ± 8.16 | 0.208 ± 0.032 |
| two_concepts / TinyCNN | spectrum_retention_1 | 0.928 ± 0.053 | 0.327 ± 0.077 | 79.08 ± 7.34 | 20.05 ± 5.69 | 87.11 ± 3.98 | 0.227 ± 0.072 |
| two_concepts / TwoLayerCNN | template_init | 0.240 ± 0.079 | 0.145 ± 0.028 | 9.26 ± 4.49 | 4.79 ± 2.30 | 98.79 ± 1.36 | 1.834 ± 0.062 |
| two_concepts / TwoLayerCNN | template_retention_1 | 0.327 ± 0.055 | 0.240 ± 0.031 | 11.39 ± 5.09 | 7.62 ± 2.80 | 98.40 ± 1.82 | 1.514 ± 0.099 |
| two_concepts / TwoLayerCNN | spectrum_retention_1 | 0.278 ± 0.140 | 0.257 ± 0.037 | 10.90 ± 10.01 | 8.96 ± 3.71 | 97.62 ± 1.55 | 1.517 ± 0.053 |

All per-run agreement, correct-pair subsets, input-effect denominators and k-dependent results are preserved in `analysis/patching_all.json`.

![Patch size dependence](figures/patch_size.png)

### 4.5 Frozen-model optimization diagnostic

The alternative head is selected by validation loss; these results do not replace the main model’s causal metrics. Gains are paired within each block.

| Setting | Frozen condition | Original head (%) | Refitted head (%) | Gain pp | Selected solver converged |
| --- | --- | --- | --- | --- | --- |
| single_shape / TinyCNN | frozen_random_unitnorm | 74.34 ± 12.68 | 96.21 ± 2.44 | 21.88 ± 11.08 | 10/10 |
| single_shape / TinyCNN | frozen_spectrum | 82.93 ± 6.90 | 99.38 ± 0.72 | 16.45 ± 7.02 | 10/10 |
| single_shape / TinyCNN | frozen_templates | 83.40 ± 7.84 | 99.77 ± 0.62 | 16.37 ± 7.80 | 10/10 |
| single_shape / TwoLayerCNN | frozen_random_unitnorm | 99.92 ± 0.16 | 100.00 ± 0.00 | 0.08 ± 0.16 | 10/10 |
| single_shape / TwoLayerCNN | frozen_spectrum | 99.96 ± 0.12 | 100.00 ± 0.00 | 0.04 ± 0.12 | 10/10 |
| single_shape / TwoLayerCNN | frozen_templates | 99.96 ± 0.12 | 100.00 ± 0.00 | 0.04 ± 0.12 | 10/10 |
| two_concepts / TinyCNN | frozen_random_unitnorm | 45.00 ± 6.64 | 71.33 ± 3.98 | 26.33 ± 6.48 | 10/10 |
| two_concepts / TinyCNN | frozen_spectrum | 47.19 ± 10.57 | 80.74 ± 4.72 | 33.55 ± 12.53 | 9/10 |
| two_concepts / TinyCNN | frozen_templates | 52.30 ± 7.57 | 81.80 ± 2.38 | 29.49 ± 8.22 | 9/10 |
| two_concepts / TwoLayerCNN | frozen_random_unitnorm | 99.02 ± 0.81 | 99.49 ± 0.58 | 0.47 ± 0.40 | 10/10 |
| two_concepts / TwoLayerCNN | frozen_spectrum | 96.68 ± 1.82 | 98.75 ± 0.75 | 2.07 ± 1.60 | 10/10 |
| two_concepts / TwoLayerCNN | frozen_templates | 96.80 ± 3.11 | 98.44 ± 1.33 | 1.64 ± 1.97 | 10/10 |

Across all conditions, 398/400 selected fits report successful solver termination. Individual gradient norms and all candidates are retained; a successful termination flag is not an information-theoretic capacity certificate.

![Frozen-head diagnostic](figures/head_optimization.png)

**Exploratory classifier sensitivity of patching.** This supplement was motivated by partial outcomes and changes only the head to its already validation-selected refit. Features and channel rankings remain fixed; these values do not replace the primary tests. Full detail for 200 evaluations is in `analysis/head_causal_diagnostic.csv`.

| Setting | Condition | Original U | Refit U | ΔU (exploratory) | Refit selected CF accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| single_shape / TinyCNN | template_init | 0.254 ± 0.061 | 0.338 ± 0.052 | 0.085 ± 0.050 | 38.11 ± 4.69 |
| single_shape / TinyCNN | template_retention_1 | 0.288 ± 0.060 | 0.432 ± 0.043 | 0.144 ± 0.060 | 51.64 ± 5.57 |
| single_shape / TinyCNN | frozen_templates | 0.252 ± 0.077 | 0.403 ± 0.037 | 0.152 ± 0.054 | 57.08 ± 2.82 |
| single_shape / TwoLayerCNN | template_init | 0.058 ± 0.040 | 0.034 ± 0.045 | -0.025 ± 0.025 | 3.84 ± 4.10 |
| single_shape / TwoLayerCNN | template_retention_1 | 0.049 ± 0.058 | 0.070 ± 0.028 | 0.021 ± 0.051 | 9.27 ± 3.84 |
| single_shape / TwoLayerCNN | frozen_templates | 0.049 ± 0.050 | 0.060 ± 0.051 | 0.012 ± 0.031 | 7.64 ± 5.09 |
| two_concepts / TinyCNN | template_init | 0.555 ± 0.092 | 0.596 ± 0.103 | 0.041 ± 0.092 | 71.97 ± 8.50 |
| two_concepts / TinyCNN | template_retention_1 | 0.580 ± 0.099 | 0.665 ± 0.108 | 0.084 ± 0.087 | 84.59 ± 10.03 |
| two_concepts / TinyCNN | frozen_templates | 0.679 ± 0.121 | 0.525 ± 0.065 | -0.154 ± 0.106 | 61.95 ± 4.06 |
| two_concepts / TwoLayerCNN | template_init | 0.095 ± 0.080 | 0.038 ± 0.057 | -0.057 ± 0.039 | 6.33 ± 2.65 |
| two_concepts / TwoLayerCNN | template_retention_1 | 0.088 ± 0.060 | 0.046 ± 0.064 | -0.042 ± 0.044 | 8.48 ± 3.37 |
| two_concepts / TwoLayerCNN | frozen_templates | 0.102 ± 0.066 | 0.060 ± 0.082 | -0.042 ± 0.053 | 11.66 ± 3.69 |

Alignment of the validation-selected channels themselves is additionally recorded in the per-run table (`selected_kernel_alignment`, `best_semantic_kernel_alignment`), without changing their selection. These are exploratory checks on using a whole-layer alignment average.

### 4.6 Matched nuisance stability

Nuisance changes alter translation and pixel noise jointly while preserving concepts. This is a separate label-stability diagnostic.

| Setting | Condition | Nuisance-image accuracy (%) | Prediction stability (%) | Mean squared probability change |
| --- | --- | --- | --- | --- |
| single_shape / TinyCNN | random_unitnorm | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.001 ± 0.000 |
| single_shape / TinyCNN | template_init | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.001 ± 0.000 |
| single_shape / TinyCNN | template_retention_1 | 99.80 ± 0.38 | 99.80 ± 0.28 | 0.002 ± 0.000 |
| single_shape / TinyCNN | frozen_templates | 83.28 ± 7.54 | 95.27 ± 2.50 | 0.000 ± 0.000 |
| single_shape / TwoLayerCNN | random_unitnorm | 99.96 ± 0.12 | 99.96 ± 0.12 | 0.000 ± 0.000 |
| single_shape / TwoLayerCNN | template_init | 99.96 ± 0.12 | 100.00 ± 0.00 | 0.000 ± 0.000 |
| single_shape / TwoLayerCNN | template_retention_1 | 99.92 ± 0.16 | 100.00 ± 0.00 | 0.001 ± 0.000 |
| single_shape / TwoLayerCNN | frozen_templates | 99.92 ± 0.16 | 99.96 ± 0.12 | 0.001 ± 0.001 |
| two_concepts / TinyCNN | random_unitnorm | 97.50 ± 1.64 | 97.62 ± 0.98 | 0.009 ± 0.003 |
| two_concepts / TinyCNN | template_init | 96.84 ± 2.18 | 97.38 ± 2.03 | 0.008 ± 0.003 |
| two_concepts / TinyCNN | template_retention_1 | 90.98 ± 8.21 | 93.95 ± 2.57 | 0.003 ± 0.001 |
| two_concepts / TinyCNN | frozen_templates | 53.12 ± 8.35 | 82.81 ± 3.42 | 0.001 ± 0.000 |
| two_concepts / TwoLayerCNN | random_unitnorm | 99.26 ± 0.87 | 99.18 ± 0.95 | 0.007 ± 0.007 |
| two_concepts / TwoLayerCNN | template_init | 99.06 ± 1.06 | 99.34 ± 0.61 | 0.008 ± 0.005 |
| two_concepts / TwoLayerCNN | template_retention_1 | 98.44 ± 1.39 | 98.67 ± 0.98 | 0.011 ± 0.005 |
| two_concepts / TwoLayerCNN | frozen_templates | 96.99 ± 2.78 | 97.46 ± 1.66 | 0.014 ± 0.006 |

Exploratory first-layer concept-versus-nuisance response magnitudes are saved in `analysis/nuisance_features.csv`. These features are not used to choose the primary channels or change training.


## 5. Interpretation

### Alignment was manipulated successfully; its causal benefit remains unresolved

The retention intervention produced a strong alignment increase in every setting. The primary U changes were +0.0345 and −0.0092 for single_shape with TinyCNN and TwoLayerCNN, respectively, and +0.0252 and −0.0074 for two_concepts. All four decisions are inconclusive under the prospective Holm-adjusted rule. The marginal interval for single_shape / TinyCNN excludes zero, but its adjusted p-value is 0.1633. It would be incorrect either to call that comparison a confirmed benefit or to describe all four marginal intervals as containing zero. The study establishes a substantial appearance change without establishing a reliable improvement in the selected-channel causal metric. It does not establish equivalence.

### Localization, identity selectivity and patching answer different questions

Stronger retention improves mean foreground localization relative to template_init in all four settings, whereas single-unit identity AUROC falls in three. For example, two_concepts / TinyCNN moves from approximately 0.977 to 0.909 AUROC while alignment increases from 0.657 to 0.946. A kernel can preserve useful spatial structure without making the validation-selected unit more discriminative for an object identity. Renderer masks avoid scoring resemblance to the template itself, but foreground overlap still cannot certify recovery of a particular local primitive.

Spectrum controls offer an additional constraint on the explanation: high predictive accuracy does not require high alignment to the corrected bank. On single_shape, spectrum_init reaches mean accuracy 100% in both architectures despite substantially lower alignment than strong template retention. This is descriptive evidence within the tested tasks. Phase scrambling preserves other structure, and these controls cannot establish that semantic geometry is the only relevant difference.

### Frozen performance is partly an optimization problem

For frozen_templates / TinyCNN, refitting only the classifier raises single_shape accuracy from 83.40% to 99.77%, with the convolutional features unchanged. The original training budget therefore substantially understates what this representation supports. On two_concepts, the same diagnostic improves accuracy from 52.30% to 81.80%. The remaining gap does not prove an information-theoretic limitation: readout class, regularization and optimization still matter. Across all conditions, 398 of 400 selected refits report successful solver termination; the other two remain documented rather than being silently treated as converged.

### Causal fidelity depends on the model being explained

A particularly instructive case is frozen_templates / TinyCNN on two_concepts: original U is approximately 0.679 despite only 52.30% accuracy. The metric rewards reproducing that model's counterfactual response, including erroneous responses. In the exploratory head-only supplement, accuracy improves to 81.80% while U falls to approximately 0.525. Features and channel rankings remain unchanged. Conversely, other fixed-feature refits increase U. Thus readout and probability behavior materially affect the operational causal score. These post hoc diagnostics motivate a controlled follow-up and do not replace the prospective tests.

### Completed audit and scope of the conclusion

The final audit covers all 400 checkpoints and 60 main dataset splits, containing 20,480 unique images. It verifies matched backgrounds, sample-weighted accuracy, frozen weights and full/no-op patch identities. Source/protocol hashes and the original first-pass report remain unchanged. An additional background audit initially used thresholded masks that omitted faint antialiasing halos. The check was corrected to reconstruct exact clean rendered support from metadata; it then passed without changing data, models or training. This was an audit-definition correction, not a result repair.

The strongest current claim is that appearance, independently measured concept properties and patching usefulness cannot be substituted for one another in this experiment. Whether retention produces a practically meaningful causal benefit remains statistically unresolved.


## 6. Limitations

1. These are two related synthetic tasks, two shallow architectures, a fixed 16-channel width and a 32×32 rendering regime. There are no natural-image experiments or large pretrained models. A result can be decisive about this manipulation and metric without establishing a universal principle.
2. The “concepts” are whole-object identities and presence indicators. They do not independently certify recovery of every local edge, corner or curvature primitive. Foreground IoU is a separate localization measurement, not proof that a unit represents the object's identity.
3. The corrected template bank is intentionally not full rank. Spectrum controls match rank, Gram matrix and Fourier magnitude but preserve other structured relationships as well; they are not guaranteed to be semantically meaningless.
4. The retention manipulation can change classification performance, confidence, feature scale, and downstream adaptation. A U difference cannot be attributed to alignment alone. Accuracy trade-offs and the spectrum controls help constrain the explanation rather than remove all confounding.
5. Channel ranking is one predeclared validation-only standardized contrast. Negative U can expose failure of that ranking, distributed coding, nonlinear interactions, or metric effects; it does not prove that no useful causal channel subset exists.
6. F reproduces the model's probability response. It can be high for an inaccurate model and can become unstable when the actual input effect is small. It is bounded above by 1 but not below. Full-patch and no-op results are identities and checks, not empirical discoveries.
7. Partial activation-map combinations may lie off the natural activation manifold, even though the source/base images are matched. Patching evidence is specific to the chosen layer, distribution and output metric.
8. Only ten independent blocks support each primary contrast. They include both data and initialization variation; these variance components are not separately estimated. Interventions within a context and treatment runs within a block are dependent. Secondary comparisons are exploratory.
9. The training budget is fixed at 160 Adam steps and not guaranteed to be converged. The linear-head refit can diagnose an optimization gap at fixed features but cannot establish a global representation ceiling, particularly for the deeper model.
10. Both tasks deliberately balance all concept states in each nuisance context. This is useful for experimental control and less representative of naturally correlated concepts. Nuisance tests jointly change translation and noise, so they do not isolate each nuisance's effect.
11. The source and protocol were frozen locally before main training, not externally preregistered. Methods were designed with knowledge of the first-pass findings. This milestone is a follow-up experiment, not a blind external replication.
12. No claim of publication novelty or general causal identifiability follows from these experiments alone. A broader literature comparison and independent replication would be required.

## 7. Next milestone, determined by the evidence

1. **Confirm a practical effect with new independent blocks.** Before collecting outcomes, choose a smallest meaningful change in U and an acceptable accuracy loss. Use the observed block variability to plan precision, accounting for the intended family of tests. Freeze a new protocol and use fresh streams. Report intervals against the practical margin, allowing benefit, harm, equivalence or inconclusive outcomes; do not extend this run until a p-value crosses a threshold.
2. **Separate readout optimization from feature learning.** Cross template_init, strong retention and matched spectrum anchors with the original training budget, a longer validation-monitored budget and a consistently fitted linear head. Apply identical head fitting to every condition and require explicit convergence diagnostics. Recompute both probability fidelity and counterfactual-label accuracy at fixed features. This directly tests whether the observed head sensitivity explains the apparent alignment–usefulness dissociation.
3. **Test primitives and distributed representations.** Add independently annotated orientation, junction and curvature interventions while holding object identity and nuisance context controlled. Compare the present unit ranking with validation-selected small subspaces under equal intervention budgets. Keep held-out interventions separate from channel/subspace selection. This can distinguish failure of a single-unit selection rule from failure of the representation.
4. **Vary rank and architecture explicitly.** Cross template geometry with multiple audited ranks and matched normalization/spectrum controls, rather than comparing only one rank-10 bank. Extend beyond the current two shallow architectures, while matching capacity and training effort where feasible. Test whether depth compensates for frozen first-layer features through downstream adaptation.
5. **Establish external relevance and novelty.** Replicate the predeclared contrast on a separate annotated image setting with defensible interventions, then compare the precise contribution against structured-filter, concept-measurement and activation-patching literature. A publishable claim should state the tested conditions and uncertainty; neither a negative headline nor a general interpretability claim is yet supported.

The next decision should be based on precision and the readout-controlled comparison. The present results justify that experiment, but do not justify claiming a decisive universal negative.


## 8. Conclusion

The completed 400-run milestone demonstrates reliable control of template alignment and a substantial fixed-feature optimization gap in frozen TinyCNN models. It does not establish a causal-usefulness improvement from stronger retention under the four corrected primary tests. Independent concept measurements and exploratory head-only interventions reveal why kernel appearance alone is an insufficient interpretation criterion. All checkpoints, measurements and rendered datasets are retained for replication; the next milestone should isolate readout effects and estimate a predeclared practically meaningful causal effect with greater precision.


## Appendix: artifacts and reproduction

`README.md` contains runnable commands. `design_freeze.json` records the prospective design and source hashes. `foundation_audit.json` verifies geometry, initialization and spectrum matching. `audit.json` verifies the completed datasets and checkpoints. `data/` contains train/validation/test images, nuisance variants, concept arrays, visible masks and context metadata. `runs/` contains every checkpoint, initial/final convolution, per-epoch history, main scalar result, intervention summaries, probe arrays and all classifier-refit candidates.

`analysis/per_run.csv` is the full run table; `summary.csv` contains means, SDs and marginal confidence intervals; `primary_contrasts.csv` contains the four Holm-adjusted prospective tests; `paired_contrasts.csv` contains exploratory paired comparisons; `head_diagnostics.csv` records the selected classifier refits; `patching_all.json` retains intervention outcomes. All scripts, execution logs, requirements and figures accompany the report. Smoke runs are clearly separated and excluded from every result table.
