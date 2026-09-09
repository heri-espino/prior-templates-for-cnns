# Validation patch-energy-matched control

Date frozen: 2026-09-09
Branch: `paper_suggestions`

## Evidential status

This is a **post-hoc robustness analysis of existing Stage B checkpoints**. It is designed after the original Stage C intervention-budget findings were known. It is not an independent replication, it does not add a new training sample, and it must not be described as confirmatory.

The purpose is narrower: determine whether the release-versus-retention comparison under selected-channel patching survives a control that is matched not only on channel count, but also on the magnitude of the activation replacement measured on validation data.

## Inputs

Read-only inputs are the epoch-200 checkpoints and validation/test data from `results/retention_release_001` for:

- tasks: `single_shape`, `two_concepts`;
- architectures: `TinyCNN`, `TwoLayerCNN`;
- blocks: 2000–2009;
- conditions: `template_retention_1`, `template_release`.

No model is retrained. No test outcome is used to select channels or controls.

## Selected rankings

Use the same three validation-only rankings as `cnn_patch_robustness`:

1. standardized concept contrast;
2. absolute AUROC deviation from 0.5;
3. singleton validation-patch fidelity.

Evaluate each at `k ∈ {1,2,4,8}`.

## Matching quantity

Let `H(x)` be the first-layer activation map immediately before the model tail. For validation matched pairs `(b,c)` belonging to intervention concept/group `g`, define the replacement energy of channel set `S` as

\[
M_g(S)=\sum_{(b,c):g}\sum_{j\in S}\lVert H_j(x_c)-H_j(x_b)\rVert_F^2.
\]

Because squared Euclidean energy is additive over channels,

\[
M_g(S)=\sum_{j\in S} e_{g,j},
\]

where `e_{g,j}` is the validation replacement energy of channel `j` for concept/group `g`.

For each selection method, concept/group and `k`, enumerate all `k`-subsets of the 16 channels. Exclude the exact selected subset. Rank candidate subsets by

\[
R_g(C,S)=\frac{|M_g(C)-M_g(S)|}{M_g(S)+\varepsilon},
\]

with `ε = 10^-12`, then by lexicographic channel tuple for deterministic tie breaking. Retain the eight closest distinct candidates. This gives eight energy-matched controls per concept/group. Candidate selection uses validation activations only.

For each control index 1–8, combine the corresponding concept-specific candidate subsets into the per-concept channel order expected by the patching routine. The remainder of each order is filled by ascending channel index and has no effect because only the first `k` channels are patched.

## Test evaluation

On test matched pairs, evaluate:

- selected fidelity `F_selected`;
- original same-size random fidelity `F_random` (for continuity with Stage C);
- validation-energy-matched fidelity `F_energy`;
- `U_random = F_selected - F_random`;
- `U_energy = F_selected - F_energy`;
- selected, random and energy-matched counterfactual accuracy;
- agreement with the actual image counterfactual prediction;
- correct-pair conditional counterfactual accuracy.

Also preserve, for every method/k/concept, the selected validation energy, each matched control energy, relative matching error, and channel indices.

## Integrity checks

- full 16-channel patch reproduces the actual counterfactual forward output within the existing tolerance;
- zero-channel patch reproduces the base output;
- every selected and matched control set has exactly `k` distinct channels;
- no energy-matched control exactly equals its corresponding selected subset;
- matching uses validation only;
- output directory is distinct from the imported Stage B evidence and original Stage C outputs;
- source, protocol, checkpoint and data hashes are recorded before evaluation;
- completed checkpoint outputs are written atomically and are resumable only under an identical design hash.

## Analysis

Use block as the paired unit (`n=10` per task/architecture/method/k). Report release-minus-retention means and marginal 95% Student-t intervals for `F_selected`, `F_random`, `F_energy`, `U_random`, `U_energy`, and the three counterfactual-accuracy quantities.

These intervals are exploratory and are not multiplicity-adjusted. The analysis must show all three rankings and all four channel budgets, not only combinations that preserve a preferred sign.

## Interpretation rule

- If a `ΔU_random` reversal remains but `ΔF_selected` does not, do **not** claim redistribution of causal information.
- If the selected-fidelity treatment contrast itself changes materially with `k`, report that as a stronger budget-dependent functional result, while still avoiding a mechanism claim.
- If the unmatched-random contrast disappears under `U_energy`, state that the original result was sensitive to intervention-magnitude matching.
- If `U_energy` retains the same qualitative budget dependence, this strengthens robustness to one specific alternative baseline; it is still exploratory and not an independent confirmation.

No outcome from this analysis may be used to rewrite this protocol.