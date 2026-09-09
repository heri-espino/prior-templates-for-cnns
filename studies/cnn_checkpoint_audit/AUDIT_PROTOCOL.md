# Independent checkpoint audit protocol

Date frozen: 2026-09-09
Branch: `paper_suggestions`

## Goal

Independently recompute a representative subset of the Stage B manuscript quantities directly from saved checkpoint weights and saved renderer data. This audit must not trust the stored scalar evaluation JSON as input to the computation being checked.

The implementation lives in a separate directory and independently reimplements the network forward structure, template bank/alignment, concept ranking, AUROC, localization and patch-fidelity calculations. Stored evaluation JSON is read **only after recomputation** to calculate discrepancies.

## Audit sample

Use the Cartesian product:

- tasks: `single_shape`, `two_concepts`;
- architectures: `TinyCNN`, `TwoLayerCNN`;
- conditions: `template_retention_1`, `template_release`;
- blocks: `2000`, `2009`;
- checkpoints: epochs `80`, `200`.

Total: 32 checkpoint evaluations.

This sample covers both endpoints of the fresh block namespace, both architectures, both tasks, both compared treatments, the release endpoint, and the final manuscript endpoint.

## Metrics recomputed independently

For each checkpoint:

1. ordinary test accuracy;
2. first-layer nearest-template alignment;
3. single-unit concept AUROC using validation-only standardized concept effects;
4. localization IoU using validation 95th-percentile channel thresholds and validation-only channel selection;
5. selected four-channel patch fidelity;
6. eight-set same-size random fidelity baseline using the recorded seed namespace;
7. `U = fidelity_selected - fidelity_random`;
8. selected four-channel counterfactual accuracy;
9. random-set counterfactual accuracy;
10. full 16-channel patch identity;
11. zero-channel patch identity.

The audit does not recompute the alternative L-BFGS head refits; those are secondary diagnostics rather than the central manuscript claim.

## Independence boundary

The audit code does **not** call `core.evaluate`, `core.collect`, `core.alignment`, `core.concept_orders`, `core.patch_eval` or `core.auc`.

It independently defines:

- the two small architectures from their documented layer specification;
- the corrected edge/corner/ring bank;
- normalization and alignment;
- renderer pair indexing;
- seed namespace used for random patch controls;
- activation collection;
- concept ranking;
- rank-based AUROC;
- localization;
- internal patching and fidelity.

It necessarily uses PyTorch to load the saved model state dict and NumPy/PyTorch to read and process the saved data.

## Comparison tolerances

The audit records raw discrepancies before applying tolerances. Default pass tolerances are:

- accuracy and counterfactual accuracy: exact to `1e-12`;
- alignment: `1e-6`;
- semantic AUROC: `1e-9`;
- localization IoU: `1e-9`;
- selected/random fidelity and U: `2e-6`;
- full/no-op maximum probability error: `2e-5`.

A tolerance failure is not automatically repaired. It must appear in the audit report with checkpoint identity and both values.

## Environment and provenance

Record:

- Python, NumPy, SciPy and PyTorch versions;
- platform;
- audit source hash;
- this protocol hash;
- every checkpoint/data SHA-256 hash;
- stored Stage B `design.json` hash.

No source result or checkpoint is modified.

## Decision rule

The manuscript may be described as checkpoint-audited only if:

1. all 32 planned recomputations complete;
2. full/no-op identities pass;
3. all central metric discrepancies are within their declared tolerances, or any exception is explained and manuscript numbers are reconciled;
4. the exact audit report and source revision are retained with the submission artifact.

This audit validates reproducibility of the sampled stored evidence. It does not establish external validity, statistical correctness of every exploratory comparison, or correctness of checkpoints not sampled.