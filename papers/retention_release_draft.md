# Template retention, learning dynamics, and causal usefulness in small convolutional networks

**Empirical draft — 8 September 2026.** Completed retention-release experiment; exploratory analysis. This draft extends the [400-run alignment study](../studies/cnn_causal_milestone/REPORT.md), without changing its historical results. The new data and code are linked below. Publication novelty has not been established.

## Abstract

Template priors may accelerate learning, but preserving their appearance could constrain subsequent adaptation. We examine this hypothesis in 200 training runs spanning two synthetic tasks, two small convolutional architectures, ten independent paired blocks, and five initialization/retention conditions. Models train for 200 epochs, with independent renderer-based concept measurements and matched activation interventions at nine checkpoints. Template initialization has an early advantage over normalization-matched random initialization on single_shape with TwoLayerCNN, but a disadvantage on two_concepts with TinyCNN. On the latter setting, template initialization reaches 99.30% mean test accuracy after 200 epochs, matching normalization-matched random initialization. Releasing retention improves final accuracy relative to constant retention by 1.21 percentage points in TinyCNN and 0.39 points in TwoLayerCNN on two_concepts. However, corresponding causal-usefulness differences have opposite signs: +0.061 and −0.058. Alignment decreases in both cases. These findings support a task- and architecture-dependent effect of retention, rather than a universal early-learning advantage or a demonstrated template representation ceiling. The statistical intervals are exploratory and marginal; no new confirmatory claim is made.

## 1. Motivation and research questions

The original project compares random initialization, template initialization and frozen templates. An initial reproduction exposed foundation issues; a subsequent controlled study corrected split streams, metric weighting and template geometry. In that 400-run study, strong retention increased kernel alignment substantially, but none of four primary causal-usefulness contrasts passed the predeclared Holm correction. Fixed-feature classifier refits also exposed an optimization gap in frozen models.

The present experiment asks three narrower questions:

1. Does template initialization accelerate learning compared with random initialization, including normalization matching?
2. Does continued retention constrain later adaptation, and does gradually releasing it improve performance?
3. When alignment changes during training, do independently measured concepts and selected-channel causal usefulness change with it?

These questions distinguish initializing with a template from forcing the model to stay close to it. “Human-readable” kernel appearance is not measured directly: there are no human ratings. Alignment is a geometric descriptor, not a substitute for semantic or causal evidence. Background literature and the earlier methodology are discussed in the [preceding paper](../studies/cnn_causal_milestone/REPORT.md); this draft makes no new claim of literature coverage or novelty.

## 2. Methods

### 2.1 Design, streams and data

The factorial design contains 2 tasks × 2 architectures × 10 blocks × 5 conditions = **200 runs**. Each model trains for 200 epochs. Blocks 2000–2009 use fresh deterministic streams, distinct from the earlier study's blocks 0–9. The master seed is 2026090617. Within a block, conditions share data, minibatch ordering and downstream initialization; block differences include both generated-data and initialization variation. The unit for paired uncertainty estimates is the block, not an image, epoch or intervention pair.

Each task/block contains 512 training, 256 validation and 256 test images. These correspond to 128, 64 and 64 independently generated nuisance contexts, respectively, each rendered in all four balanced states. Images are 32×32 grayscale. Stored data include labels, renderer concept indicators, visible masks, matched nuisance variants and context metadata. All splits use separate seed namespaces. The resulting experiment contains 20,480 main images shared across model conditions, plus nuisance variants.

**single_shape** classifies line, circle, triangle and square. Every context renders all four identities under the same nuisance realization. **two_concepts** renders two objects: square/circle and line/triangle. The label is circle_present + 2×triangle_present. Object positions are swapped independently across contexts; single-bit interventions alter one identity while retaining the other.

Rendering varies angle from 0 to π/4, width from 0.8 to 1.6, scale from 0.85 to 1.15, translation, and Gaussian noise with standard deviation sampled between 0 and 0.05. A 5×5 occluder occurs with probability 0.1. Threefold antialiasing is used. Matched counterfactual images share nuisance context; separate nuisance images change translation and noise while preserving identities and the other sampled settings. This controlled distribution is not a natural-image benchmark.

### 2.2 Models and priors

**TinyCNN** uses a 1→16 convolution with 9×9 kernels, padding 4 and no bias, followed by ReLU, global max pooling and a four-class linear head (1,364 parameters). **TwoLayerCNN** adds a trainable 16→16 convolution with 3×3 kernels and ReLU before pooling (3,668 parameters). Both are small architectures; the comparison isolates added depth within this family, not broad architectural generalization.

The corrected prior contains 8 edge, 4 corner and 4 ring filters. Kernels are centered and normalized to unit norm. The bank has rank 10: genuine two-ray corners correct a flaw in the original archive, but edge redundancies remain. The current experiment does not include the previous study's spectrum controls, so it cannot isolate every geometric confound in this new schedule comparison.

| Condition | Initialization | Retention schedule |
|---|---|---|
| random | Default PyTorch random | λ=0 |
| random_unitnorm | Same random draw, centered and unit norm | λ=0 |
| template_init | Corrected template bank | λ=0 |
| template_retention_1 | Corrected template bank | λ=1 throughout |
| template_release | Corrected template bank | λ=1 through epoch 10, decreasing linearly to 0 at epoch 80 |

All layers are trainable. Unlike the earlier frozen-model experiment, there is no frozen condition here. The release schedule is λ(e)=clip((80−e)/70,0,1). Release and constant-retention histories match exactly through epoch 10 in the saved measurements. Early differences between those two conditions therefore cannot be evidence for a release effect.

### 2.3 Optimization and recording

The loss is cross-entropy plus λ‖W−W_anchor‖²_F/‖W_anchor‖²_F, applied only to the first convolution. All template anchors have squared norm 16. Adam uses learning rate 0.003 and batch size 128; four optimizer steps per epoch give 800 steps per model. There is no learning-rate scheduler, early stopping or checkpoint selection. The final-epoch outcome is reported as final performance; 200 epochs is not assumed to guarantee convergence.

Every epoch records sample-weighted training loss/accuracy, validation loss/accuracy, retention coefficient and penalty, kernel alignment/norm, first-layer gradient norm and pooled feature norm. Training metrics aggregate predictions made during parameter updates; validation metrics evaluate the completed epoch. Saved checkpoints occur at initialization and epochs 1, 5, 10, 20, 40, 80, 120, 160 and 200. Model, optimizer and random state are retained for resumption.

The recorded environment is Python 3.11.16, PyTorch 2.14.0+cu130, NumPy 2.4.6 and Linux/WSL2, with two CPU threads. Despite the CUDA-enabled package name, the provided runner keeps tensors and models on CPU. Cross-study numerical comparisons can also reflect environment differences; the paired comparisons within this run are the principal evidence.

### 2.4 Independent concept and causal measurements

**Alignment** averages, over the 16 first-layer kernels, the maximum signed centered cosine similarity to the corrected bank. Multiple learned kernels may select the same template.

**Concept AUROC** measures held-out identity/presence discrimination by a single first-layer unit per concept. Channel and sign are chosen using validation-only standardized positive–negative activation contrasts. **Localization IoU** separately selects the validation-best foreground channel, using each channel's validation 95th-percentile activation threshold, and evaluates visible renderer masks on concept-positive test images. Localization channels can differ from AUROC channels. Neither metric selects channels by template resemblance.

**Matched patching** copies selected first-layer activation maps from a counterfactual image into its matched base image and propagates through the unchanged network tail. All 12 directed identity substitutions per test context are used for single_shape (768 pairs); two_concepts uses the eight directed one-bit changes (512 pairs). Validation concept contrasts select channels. The primary displayed patch size is four channels, compared with eight fixed random rankings of the same size; sizes one and eight are also retained.

For base probability p₀, actual image-counterfactual probability p₁ and internally patched probability pS, fidelity is

\[
F=1-\frac{\sum\|p_S-p_1\|_2^2}{\sum\|p_1-p_0\|_2^2},\qquad
U=F_{selected,4}-\operatorname{mean}(F_{random,4}).
\]

No-op fidelity is zero and full-channel fidelity is one when the denominator is nonzero. F is not a mediated fraction and is unbounded below. U measures a selected-set advantage over random sets, not all causal information in the representation. Ground-truth counterfactual accuracy is reported separately because fidelity can reproduce an incorrect model response. Alternative linear-head refits use fixed features, three regularization values and validation-loss selection; they do not replace main causal metrics. All 1,800 selected refits report successful solver termination, which is not a proof of global representation capacity.

### 2.5 Analysis and integrity

We analyze early mean validation accuracy over epochs 1–10, first epoch reaching 95% validation accuracy, final test outcomes and final-20-epoch validation-loss slopes. First threshold attainment can be transient and is not a sustained-convergence measure. All 200 runs reached the threshold, so these particular means have no non-attainment censoring.

Paired differences use ten blocks per setting. We provide marginal 95% Student-t intervals, assuming reasonably behaved block differences. The release schedule and training design were saved before execution; the present contrasts, displays and statistical analysis were assembled after outcomes were available. They are **exploratory**, not a newly preregistered confirmatory test family. Intervals are not adjusted for the many outcomes and contrasts. We do not promote isolated intervals excluding zero into confirmatory claims. Curves display means ± standard deviations across blocks; epochs are not independent replicates.

The upload manifest verified 12,129 files with no mismatches, and core/experiment hashes matched the supplied source. The present analysis independently checked the published tables against all raw histories and scalar evaluation JSON files: 40,000 epoch records and 1,800 checkpoint evaluations. Stored full/no-op identities and defined fidelity denominators passed checks. This is artifact and numerical-consistency validation; it does not independently rerun every saved model's forward pass. Raw imported evidence is unchanged.

## 3. Results

### 3.1 Early template advantages depend on the setting

| Task / model | random | random_unitnorm | template_init | release / constant retention |
|---|---:|---:|---:|---:|
| single_shape / TinyCNN | 79.46% | 80.09% | 76.23% | 75.69% |
| single_shape / TwoLayerCNN | 79.29% | 84.21% | 87.55% | 87.52% |
| two_concepts / TinyCNN | 64.10% | 61.21% | 54.37% | 52.37% |
| two_concepts / TwoLayerCNN | 71.13% | 74.41% | 72.72% | 71.90% |

Values are mean validation accuracy averaged over epochs 1–10. Template_init minus random_unitnorm is +3.34 percentage points [1.25, 5.42] for single_shape / TwoLayerCNN, but −6.84 [−10.26, −3.43] for two_concepts / TinyCNN. The other marginal intervals include zero. Thus even the direction of the early difference depends on task and architecture. The normalized random control reduces the apparent benefit relative to default random.

![Early learning](../analysis/retention_release_001/early_learning.png)

Time-to-threshold and early area are not interchangeable. On single_shape / TinyCNN, template_init has a lower early average but reaches 95% at mean epoch 7.7, compared with 8.2 for random_unitnorm. On two_concepts / TinyCNN, release reaches 95% at epoch 46.5 on average, versus 66.8 for constant retention, 23.4 for template_init and 25.5 for random_unitnorm. Release improves the constrained model's threshold time, but does not make it the fastest learner.

### 3.2 Longer training removes much of the apparent shallow-model deficit

On two_concepts / TinyCNN, template_init rises from 97.66% test accuracy at epoch 40 to 99.30% at epoch 200, matching random_unitnorm's final 99.30%. Constant retention also improves substantially, from 92.54% to 98.01%. The initial short-budget deficit therefore did not establish that templates were stuck at a fixed representational ceiling. The constrained model still improves late: its mean final-20-epoch validation-loss slope is negative. Continued improvement does not guarantee eventual equality, but it precludes treating the observed training horizon as a proven asymptote.

![Learning, loss and alignment](../analysis/retention_release_001/learning.png)

### 3.3 Release improves compositional accuracy but sacrifices alignment

| Setting | Constant retention accuracy | Release accuracy | Paired gain, pp [95% interval] | Alignment: constant → release |
|---|---:|---:|---:|---:|
| single_shape / TinyCNN | 99.96% | 100.00% | +0.04 [−0.05, 0.13] | .959 → .690 |
| single_shape / TwoLayerCNN | 99.92% | 99.96% | +0.04 [−0.05, 0.13] | 1.000 → .942 |
| two_concepts / TinyCNN | 98.01% | 99.22% | +1.21 [0.71, 1.71] | .947 → .580 |
| two_concepts / TwoLayerCNN | 99.14% | 99.53% | +0.39 [0.04, 0.74] | .999 → .907 |

Release reduces mean alignment in all four settings. It retains more alignment than template_init in each setting, indicating dependence on the optimization path after λ reaches zero. This does not isolate alignment as the cause of any accuracy difference; the schedule changes the learned representation and downstream adaptation jointly.

![Paired release effects](../analysis/retention_release_001/release_effects.png)

### 3.4 Independent concept and causal outcomes do not move uniformly

| Setting | AUROC constant → release | Localization IoU constant → release | U constant → release | Paired ΔU [95% interval] |
|---|---:|---:|---:|---:|
| single_shape / TinyCNN | .938 → .975 | .544 → .438 | .358 → .335 | −.022 [−.087, .043] |
| single_shape / TwoLayerCNN | .874 → .846 | .566 → .505 | .067 → .060 | −.007 [−.052, .039] |
| two_concepts / TinyCNN | .946 → .984 | .347 → .235 | .657 → .718 | +.061 [.027, .095] |
| two_concepts / TwoLayerCNN | .776 → .771 | .391 → .363 | .061 → .003 | −.058 [−.094, −.021] |

In the compositional task, reduced retention is associated with higher predictive accuracy in both architectures, but opposite selected-channel U differences. In TinyCNN, reduced alignment accompanies better single-unit concept discrimination and a higher U; in TwoLayerCNN, the U advantage largely disappears. Localization falls in all four settings. Object foreground overlap, identity selectivity and selected-set patching usefulness therefore cannot be substituted for each other.

The distinction between relative fidelity and intervention correctness is visible even within TinyCNN: on two_concepts, release has U=.718 versus .657 for constant retention, while selected-patch counterfactual accuracy is **90.64% versus 93.61%**. A higher advantage over random patches need not mean higher absolute intervention accuracy. In TwoLayerCNN, selected counterfactual accuracy is only 4.02% for release despite 99.53% ordinary accuracy; four first-layer channels often fail to enact the intended change. This could reflect distributed coding, the channel-ranking rule or downstream nonlinear interaction, rather than absence of causal information.

![Concept and causal trajectories](../analysis/retention_release_001/concepts.png)

## 4. Interpretation

The owner's motivating hypothesis receives partial, conditional support. Templates can provide an early advantage, as in single_shape / TwoLayerCNN, but do not generally do so. Template initialization alone catches up on the shallow compositional task. Constant retention leaves a residual fixed-budget cost, and release alleviates that cost. Thus initialization and persistent constraint must be distinguished before attributing late performance to templates themselves.

The strongest mechanistic observation is an architecture-dependent dissociation: release lowers alignment and improves compositional prediction in both architectures, but its selected-channel causal-usefulness difference changes sign with depth. The current experiment does not determine whether this difference arises from semantic coding, channel selection, patch distribution shift or readout behavior. It does show why one metric cannot stand in for the others.

These findings do not supersede the previous study's four inconclusive corrected tests: this experiment uses fresh blocks, a different horizon and a new treatment comparison. They add longitudinal evidence and a release intervention, with exploratory uncertainty estimates. A convincing eventual paper would state these conditional findings and replicate the key interaction under a prospectively fixed analysis.

## 5. Limitations

- Only two related synthetic tasks, two small architectures and one template bank are covered. No natural-image or human-readability validation is present.
- Conditions share balanced rendered contexts. This improves control but simplifies real concept correlations. Whole-object concepts do not independently certify every local primitive represented in the prior.
- A single release schedule, optimizer and horizon were tested. No claim of optimal scheduling or convergence follows. There are only ten independent blocks per setting.
- Repeated checkpoint test evaluation was scheduled and does not alter training, but subsequent schedule development using these curves would require new held-out evidence.
- Exploration spans many outcomes and contrasts. Marginal intervals do not control familywise error, and no practical-equivalence margin was fixed for this analysis.
- Patching may generate unnatural combinations of feature maps. Its fidelity is tied to the model's probability outputs and the chosen ranking/layer. Full/no-op identities are checks, not discoveries.
- Unit selection is refreshed from validation at each checkpoint. A metric trajectory can include changes in selected units, not just changes in one persistent feature. Stored probe arrays permit future unit-tracking analyses.
- The current result audit checks stored evidence rather than independently reproducing all model evaluations. Solver termination flags are not guarantees of representation optimality.
- Broader normalization/rank/spectrum controls from the earlier study were not crossed with the release schedule. Runtime comparisons are not analyzed.

## 6. Next experiments

1. **Confirm the architecture interaction.** Fix the release-versus-constant comparison in two_concepts and a practical accuracy/U margin before generating new blocks. Estimate the task/depth interaction with paired structure and a declared multiplicity plan.
2. **Separate ranking from distributed causal information.** Compare validation-selected channel sets with equal-budget subspaces and test multiple patch sizes. Track unit identities over training; use fresh evaluation data for any new selection procedure.
3. **Control the readout.** Apply consistently optimized or refitted heads at fixed checkpoints, retaining identical concept rankings and pairs, then compare probability fidelity with counterfactual-label accuracy. Existing refit artifacts support preliminary diagnostics but not a retroactive confirmatory claim.
4. **Test schedule specificity and external relevance.** Prospectively vary release timing and anchor strength with matched spectrum controls, then replicate on a separate annotated task. Do not select an optimal schedule from the current test curves and report it as independently validated.

## 7. Conclusion

Template initialization does not uniformly accelerate learning, and a short-budget deficit does not establish that templates are stuck. Longer training lets the shallow compositional template-initialized model match normalization-matched random accuracy. Releasing retention improves on a constant constraint while preserving some initial-template alignment, but its causal-usefulness effect depends on architecture. The evidence favors separating learning speed, retained appearance, concept measurements and intervention behavior rather than combining them into a single interpretability claim.

## Reproducibility and artifacts

- [Original uploaded result provenance](../results/retention_release_001/README.md) and [saved design](../results/retention_release_001/design.json).
- [Runner and evaluation code](../studies/cnn_release_experiment/README.md).
- [Analysis script](../analysis/retention_release_001/analyze.py), [consistency audit](../analysis/retention_release_001/audit.json), [per-run endpoints](../analysis/retention_release_001/endpoints.csv), [means/SDs](../analysis/retention_release_001/summary.csv), and [paired marginal intervals](../analysis/retention_release_001/paired_contrasts.csv).
- From the repository root, run `python analysis/retention_release_001/analyze.py` in an environment with the study dependencies. It reads imported evidence and writes separate derived outputs, leaving the upload unchanged. Narrative text is reviewed prose and is not automatically rewritten by the plotting script.
