# Submission revision plan

Branch: `paper_suggestions`
Date: 2026-09-09

This plan turns the scientific review in `PAPER_SUGGESTIONS.md` into an auditable sequence of changes. Each stage has a distinct evidential role. Existing exploratory analyses are not retroactively relabeled as confirmatory.

## Stage 0 — freeze claims and preserve provenance

- Keep Stage A prospective tests unchanged.
- Keep Stage B/C exploratory labels unchanged.
- Do not overwrite original robustness outputs or protocols.
- Treat every new reanalysis of the existing Stage B checkpoints/test data as exploratory.
- Use new output directories for new controls.

**Done when:** manuscript, code and review notes agree on which results are prospective, exploratory, or future confirmatory.

## Stage 1 — decompose the custom usefulness score

The current headline quantity is

\[
U(k)=F_{\mathrm{selected}}(k)-F_{\mathrm{random}}(k).
\]

The existing GPU robustness JSON already stores `fidelity` and `random_fidelity` separately for each ranking and channel budget. Extend `analysis/patch_robustness_gpu_001/review.py` to report paired release-minus-retention contrasts for:

1. selected fidelity `fidelity`;
2. random baseline fidelity `random_fidelity`;
3. their difference `causal_usefulness`;
4. selected counterfactual accuracy `cf_accuracy`;
5. random counterfactual accuracy `random_cf_accuracy`.

Add a component-level table/figure so a reader can tell whether a sign change in \(\Delta U\) is driven by selected patches, the random baseline, or both.

**No new training. No new forward passes.**

**Done when:** the regenerated robustness report exposes the components of \(U\), not only the difference.

## Stage 2 — add a validation patch-energy-matched control

The current random baseline is matched on channel count but not on intervention magnitude. Add a separate post-hoc checkpoint evaluation that matches control subsets to the selected subset's validation-only activation replacement energy.

For concept/group \(g\) and channel set \(S\), define

\[
M_g(S)=\sum_{(b,c):g}\sum_{j\in S}\lVert H_j(x_c)-H_j(x_b)\rVert_F^2.
\]

For every selection method and \(k\in\{1,2,4,8\}\), choose control channel sets using validation data only whose \(M_g\) is closest to the selected set. Evaluate those sets once on test. Preserve the original unmatched-random baseline alongside the matched control.

Report matching quality, selected fidelity, matched-control fidelity, their difference, and absolute counterfactual accuracy.

**No retraining. Existing checkpoints only.**

**Done when:** the retention/release comparison has been tested against a same-size, validation-energy-matched baseline in a new output tree.

## Stage 3 — update the manuscript only after Stages 1–2 are numerically reviewed

Revise the title/front matter toward the narrower demonstrated claim:

> Template Priors in Small CNNs: Kernel Alignment, Learning Dynamics, and Intervention-Budget Sensitivity

Then update:

- abstract;
- contribution paragraph;
- matched-patching methods;
- Stage C results;
- interpretation;
- limitations;
- conclusion;
- supplementary component tables.

Rules:

- If the decomposition shows that \(\Delta U\) is mainly baseline-driven, say so explicitly.
- If selected fidelity itself reverses, distinguish that stronger finding from \(\Delta U\).
- If the energy-matched control removes the effect, narrow the claim rather than hiding the result.
- Do not call the matched-control analysis confirmatory.

**Done when:** every headline sentence can be traced to a displayed estimand and its evidential status.

## Stage 4 — independent checkpoint audit

Before journal sign-off, independently reproduce a representative subset from saved checkpoints rather than trusting stored scalar JSON alone.

Minimum audit set:

- both architectures;
- both Stage B treatments (`template_retention_1`, `template_release`);
- both tasks;
- at least blocks 2000 and 2009;
- epoch 200, plus one intermediate checkpoint for trajectory validation;
- ordinary accuracy, alignment, selected/random fidelity, counterfactual accuracy, and full/no-op identities.

Use a fresh script/environment path and compare against stored results with explicit tolerances. Record software versions, hashes and all discrepancies.

**Done when:** a signed audit report lists what was recomputed, numerical deviations, and whether any manuscript value requires correction.

## Stage 5 — freeze a prospective independent confirmation protocol

Only after the exploratory decomposition and matched-control results are known, write—but do not alter after outcome inspection—a fresh protocol for the narrow question that survived.

Default target if the current pattern survives:

- task: `two_concepts` plus one additional controlled/annotated task if feasible;
- architectures: TinyCNN and TwoLayerCNN;
- treatments: constant retention vs release;
- fresh block namespace;
- \(k=1,2,4,8\) fixed in advance;
- primary estimands include selected fidelity, energy-matched-control fidelity, absolute counterfactual accuracy, and treatment-by-budget interaction;
- multiplicity and interval policy fixed before execution.

Do not select only the channel budget that gives the preferred sign.

**Done when:** protocol hash/commit predates all new outcome files.

## Stage 6 — run the fresh confirmation

Train/evaluate only the design frozen in Stage 5. Keep discovery and confirmation results separate in tables and prose.

Possible outcomes:

- **replicates:** promote intervention-budget sensitivity to the main empirical contribution;
- **partially replicates:** state the restricted architecture/task dependence;
- **fails:** retain the original exploratory result as a documented measurement-sensitivity observation and narrow the paper.

## Stage 7 — submission artifact and anonymous package

After the audit and final numbers are frozen:

- build the anonymous TMLR PDF;
- create an anonymized code/checkpoint package without identifying Git metadata or private handoff files;
- include exact reproduction commands, dependencies, hashes and expected outputs;
- verify the package from a clean path/environment;
- keep the public GitHub/Zenodo citation out of the anonymous review PDF;
- archive the accepted/final public snapshot later with real author metadata and DOI.

## Priority order

1. **Stage 1:** decompose \(U\) from existing JSON.
2. **Stage 2:** energy-matched checkpoint control.
3. **Stage 4:** independent audit.
4. **Stage 3:** final manuscript rewrite based on actual new evidence.
5. **Stages 5–6:** fresh prospective confirmation.
6. **Stage 7:** anonymous submission package.

The paper should not acquire broader claims merely because additional analyses are run. The purpose of these stages is to make the existing central result easier to falsify and, if it survives, more credible.