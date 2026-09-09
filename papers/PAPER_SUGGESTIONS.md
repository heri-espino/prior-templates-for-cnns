# Submission-oriented paper review and revision suggestions

Date: 9 September 2026
Branch: `paper_suggestions`
Target manuscript: `papers/main.tex`

## Verdict

**There is a real paper here, and the central empirical observation is interesting.** The strongest contribution is not template initialization itself, nor the general fact that activation patching is measurement-sensitive. Those are established. The interesting contribution is the controlled intersection of (i) a trainable low-level template prior, (ii) retention versus release during optimization, and (iii) the finding that the retention–release comparison changes with the intervention budget even when the morphology contrast survives one-to-one kernel matching.

**I would not submit the current snapshot yet.** It is close to a credible TMLR submission, but two submission blockers remain in the repository itself: the independent checkpoint-based reproducibility audit is unfinished, and an anonymous code/checkpoint package has not been assembled. Scientifically, the most important additional analysis is to decompose the custom causal-usefulness metric `U` into its selected and random components across intervention sizes and to add a control matched on patch magnitude or activation variance. A fresh, pre-specified confirmation of the intervention-budget curve would turn the most interesting result from exploratory evidence into a substantially stronger contribution.

My practical rating of the current version:

- **Interest / research question:** strong.
- **Novelty of ingredients:** modest; appropriately acknowledged.
- **Novelty of the combination and empirical finding:** credible and potentially publishable.
- **Statistical transparency:** strong.
- **External validity:** limited.
- **Causal-identification strength:** intentionally limited and correctly caveated.
- **Submission readiness:** not yet; audit + artifact anonymization are hard blockers, and the main metric needs one more robustness layer.

## What is already strong

### 1. The paper has a defensible contribution boundary

The manuscript correctly avoids claiming novelty for Gabor/template initialization, fixed spatial filters, concept-based analysis, activation patching, or the general observation that patching choices matter. This is important because prior work already covers each ingredient separately. The contribution is instead framed as a controlled empirical extension at their intersection.

That is the right positioning. Do not broaden it.

### 2. The inferential bookkeeping is unusually careful

The paper distinguishes:

- Stage A: four locally prospective, Holm-adjusted tests;
- Stage B: a longer fresh-block retention/release experiment whose displayed contrasts are exploratory;
- Stage C: a post-hoc reanalysis of existing checkpoints and test data;
- the kernel-matching follow-up: derived evidence from saved checkpoints, not another replication.

This prevents the common mistake of treating hundreds of trained models, checkpoints, images, or interventions as hundreds of independent experimental units. Keep this structure.

### 3. The negative result is informative rather than merely null

Stage A strongly manipulates kernel alignment but does not establish a corrected improvement in the four primary `U` tests. That is useful because it rejects the simple inference

> more template-looking kernels ⇒ a demonstrated selected-channel intervention advantage.

The paper then does something better than stopping at the null result: it investigates optimization horizon, release, channel selection, and patch size.

### 4. The one-to-one matching follow-up closes a real loophole

Nearest-template alignment can overstate bank coverage because several learned kernels can reuse the same template. The assignment-based analysis directly addresses this. The retained-kernel contrast surviving one-to-one matching makes the morphology result substantially more convincing.

### 5. The manuscript is appropriately cautious about interpretability

It does not equate a visually recognizable kernel with human interpretability, a concept score with causal use, or activation patching with a unique mechanism. This is consistent with current work on concept entanglement, encoding-versus-influence, circuit faithfulness, and mechanistic non-identifiability.

## Main scientific weaknesses, in priority order

### Critical 1 — `U` is a difference of two size-dependent quantities

The paper defines

`U(k) = F_selected(k) - mean(F_random(k))`.

The headline Stage C observation is a sign reversal in the release-minus-retention difference of `U` as `k` changes. But the manuscript itself correctly notes that the random baseline also changes with `k`. Therefore, a sign reversal in `ΔU(k)` does **not** by itself show that useful causal information becomes more or less concentrated.

Before submission, add plots/tables for, at minimum:

1. `F_selected(k)` by condition;
2. `F_random(k)` by condition;
3. release-minus-retention differences for each component;
4. absolute selected-patch counterfactual accuracy by `k`;
5. ideally, a direct effect-size curve for the actual counterfactual probability/logit response being reconstructed.

A reviewer should be able to see whether the `ΔU` reversal is driven by the numerator of the scientific story (`F_selected`), the baseline, or both.

**This analysis can likely be performed from the existing checkpoint evaluations and is higher priority than training a larger model.**

### Critical 2 — the random baseline is not matched on intervention magnitude

The limitations section states that random channel sets are not matched on activation variance or patch magnitude. This is the most obvious control gap because selected channels can systematically receive larger perturbations than random channels.

Add a matched baseline such as one of the following:

- sample random channel sets whose activation-change norm is matched to the selected set;
- variance-match replacement activations per channel;
- stratify random sets by patch magnitude and compare within strata;
- use a permutation/control intervention that preserves the distribution of patch magnitudes.

The exact design should be fixed before inspecting the resulting treatment contrast. If the release/retention ordering survives this control, the result becomes much harder to dismiss as a perturbation-scale artifact.

### Critical 3 — the most interesting result is exploratory

The intervention-budget reversal is the paper's strongest reader-facing finding, but Stage C reuses Stage B checkpoints and test data after the original findings were known. The manuscript is transparent about this, which is good, but a skeptical reviewer can still say: "the headline finding is post hoc."

The cleanest upgrade is a **small independent confirmation**, not a huge benchmark campaign:

- fresh block namespace;
- pre-specify the full `k ∈ {1,2,4,8}` curve;
- pre-specify all ranking rules to be tested;
- pre-specify selected fidelity, matched-random fidelity, absolute counterfactual accuracy, and the primary architecture × treatment × budget interaction;
- pre-specify multiplicity handling;
- do not tune the release schedule on the confirmation data.

Even one focused fresh experiment on the compositional task would materially improve the paper.

### Critical 4 — finish the reproducibility audit before submission

This is already documented as unfinished. Treat it as a hard gate. The audit should independently recompute a representative set of checkpoints and metrics from model files, not merely validate stored summaries. At minimum, verify:

- ordinary predictions/accuracy;
- template alignment and one-to-one assignment;
- concept-ranking selection from validation only;
- selected and control patching metrics;
- no-op/full-patch identities;
- exact release schedule and retained anchors;
- the figures/tables generated from raw results.

The paper should only claim reproducibility that has actually been checked.

## Important but non-blocking improvements

### 1. Change the title

Current title:

> Template Priors in Small CNNs: Learning Dynamics and the Measurement Dependence of Causal Usefulness

Suggested title:

> **Template Priors in Small CNNs: Kernel Alignment, Learning Dynamics, and Intervention-Budget Sensitivity**

Why: `causal usefulness` is a custom operational term tied to `U`, while the demonstrated result is more precisely a dependence on selection and intervention budget. The suggested title is both stronger and safer.

### 2. Simplify the abstract

The current abstract is accurate but overloaded with counts (400 runs, 200 runs, 80 checkpoints, 2,000 states) and several caveats. Raw run counts can also be mistaken for independent replication even though the paper correctly explains that they are not.

Prefer the conceptual sequence:

1. morphology ≠ functional use;
2. retention manipulates morphology but does not establish a primary patching benefit;
3. release improves constrained-model accuracy while reducing alignment;
4. the release/retention patching comparison reverses across budgets;
5. one-to-one matching preserves the morphology contrast;
6. therefore conclusions depend on the operationalization of use.

A proposed replacement is in `SUGGESTED_FRONT_MATTER.tex`.

### 3. Reduce defensive repetition in the main text

The caution is scientifically good, but the manuscript repeats variants of:

- "does not establish";
- "not independent replication";
- "exploratory";
- "not human interpretability";
- "not causal identifiability".

Keep these qualifications where they change interpretation, but consolidate repeated wording into the inferential-role section and Limitations. The paper currently risks making the contribution sound weaker than the data warrant.

### 4. Shorten the related-work boundary table in the main body

The source-by-source comparison is useful for internal rigor, but the full defensive table may be more detail than a reader needs. Consider moving the detailed table to an appendix and leaving a compact main-text paragraph organized around three lines:

- structured/fixed filters;
- concepts/representation;
- causal patching/faithfulness.

The key sentence should be explicit: the paper studies whether a training-time morphology constraint changes independently measured function, and shows that the answer depends on the intervention budget.

### 5. Replace raw model-count rhetoric with experimental-unit language

When possible, say "10 paired blocks across five conditions" or "a 10-block factorial experiment" rather than leading with "200 runs". The raw number of trained models is useful for reproducibility, but the scientific replication unit is the block.

### 6. Make the claim ladder explicit

I recommend a short paragraph near the end of the Introduction:

- **Established within the tested setting:** retention changes kernel alignment; release can improve constrained-model compositional accuracy; the morphology contrast persists under assignment matching.
- **Exploratory:** the relative selected-channel intervention score changes sign with intervention budget.
- **Not established:** that release redistributes causal information, that templates improve human interpretability, or that the measured patching explanation is unique.

This would make reviewer interpretation almost automatic.

### 7. Show a morphology–function scatter, but do not overinterpret it

For the 10 blocks per condition, plot assignment alignment against:

- accuracy;
- single-unit concept AUROC;
- localization IoU;
- selected fidelity;
- absolute counterfactual accuracy.

Use this descriptively. The point is not to claim absence of correlation from `n=10`, but to visually demonstrate why morphology and function should not be collapsed into one score.

### 8. Clarify the role of single-unit concept AUROC

The current concept metric asks whether one first-layer unit discriminates a renderer concept. That is a legitimate unit-selectivity metric, but it is not a complete test of whether the representation contains the concept. A multichannel linear probe could show whether information exists even when no single unit is selective.

This is optional for the current paper, but if added, label the distinction clearly:

- **single-unit concept selectivity** versus
- **distributed concept decodability**.

This would connect naturally to the intervention-budget story.

### 9. Natural-image experiments are useful, but not the first thing I would add

A reviewer may ask about external validity because the paper uses two related synthetic tasks and two tiny architectures. A natural-image benchmark would help, but it is not the most efficient next experiment. First close the internal causal-measurement loopholes and independently confirm the budget result. After that, a small natural-image or larger-CNN extension would be valuable evidence of scope.

## Suggested reviewer-facing interpretation

### Thesis

A structured first-layer prior can make kernels retain recognizable morphology, and releasing the constraint can improve predictive optimization while preserving part of the initialization path.

### Antithesis

Morphology is not function. Unit selectivity, localization, selected-channel patch fidelity, and absolute counterfactual correctness can disagree; moreover, a relative patching comparison can reverse when the number of intervened channels changes.

### Synthesis

The scientifically useful result is not "templates are interpretable" or "release helps causal usefulness." It is:

> **Training-time control of kernel morphology changes learning trajectories, but the functional interpretation of that change is not invariant to the intervention design.**

That is a solid TMLR-style empirical lesson if the evidence is fully audited and the main patching metric is decomposed/controlled.

## Submission checklist

### Must complete before submission

- [ ] Independent checkpoint-level reproducibility audit.
- [ ] Anonymous code/checkpoint/supporting-material archive.
- [ ] Confirm all bibliography metadata and final author/declaration requirements.
- [ ] Decompose `U` into selected and random fidelity across `k`.
- [ ] Add a patch-magnitude or activation-variance-matched control, or explicitly decide to submit without it and expect reviewer pressure on this point.
- [ ] Human scientific read by all authors/coauthors.

### Strongly recommended

- [ ] Fresh pre-specified confirmation of the budget curve.
- [ ] Retitle around intervention-budget sensitivity rather than generic "causal usefulness."
- [ ] Shorten abstract and related-work boundary.
- [ ] Move some repeated caveats to Limitations.
- [ ] Show component-wise `F_selected`, `F_random`, and absolute counterfactual-accuracy curves.

### Optional extensions

- [ ] Distributed concept probe.
- [ ] Additional architecture or natural-image task.
- [ ] Alternative template bank / rank-matched structured control under the release schedule.
- [ ] Logit-difference or divergence-based patching metric in addition to probability-space fidelity.

## Bottom line

If submitted **today**, I would expect serious but addressable reviewer questions about the post-hoc status of the headline reversal, the custom difference metric `U`, and unmatched random interventions. I would not expect the work to be dismissed as uninteresting or as merely reproducing Gabor initialization: the retention/release × measurement-budget interaction is a distinct empirical story.

If the audit is completed, the artifact is anonymized, the `U` components are exposed, and one matched intervention control is added, I would consider the manuscript **submission-ready for TMLR with appropriately scoped claims**. A fresh confirmatory budget experiment would move it from "credible exploratory study" to a much stronger paper.
