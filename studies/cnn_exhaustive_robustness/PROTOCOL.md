# Exhaustive robustness map: retention strength, release timing, and intervention budget

Protocol frozen: 2026-09-09
Branch: `paper_suggestions`

## Statistical status

This is a **separate robustness study** written after Stages A--C and after the frozen 20-block prospective confirmation protocol. It does not replace, enlarge, or redefine the primary confirmation on blocks 4000--4019.

No outcome from blocks 5000--5049 may be inspected before this protocol commit exists. All analyses here are secondary to the prospective confirmation, even when confidence intervals or corrected p-values are reported.

## Fresh sample

Use deterministic renderer blocks **5000--5049** (50 fresh blocks).

Cross:

- tasks: `single_shape`, `two_concepts`;
- architectures: `TinyCNN`, `TwoLayerCNN`;
- six template-prior profiles;
- 200 epochs;
- same optimizer, learning rate, batch size, renderer and template bank as Stage B;
- final epoch 200 is the robustness checkpoint.

Total trained models:

\[
50\times 2\times 2\times 6=1200.
\]

The 50-block sample size is fixed before outcomes and is not adapted to interim results.

## Template-prior profiles

All profiles begin from the same template-initialized first layer within a block/task/architecture pair and use the same deterministic minibatch order. Let `lambda(e)` denote the normalized template-retention coefficient at epoch `e`.

1. `template_init`: `lambda(e)=0` for all epochs (immediate release after initialization).
2. `retention_0p1`: `lambda(e)=0.1` for all epochs.
3. `retention_1`: `lambda(e)=1` for all epochs.
4. `release_early`: linear release from 1 to 0 over epochs 5--40.
5. `release_default`: linear release from 1 to 0 over epochs 10--80 (the Stage-B schedule).
6. `release_late`: linear release from 1 to 0 over epochs 40--160.

No profile is removed after seeing outcomes.

## Intervention measurements

At epoch 200, use the same three validation-only channel rankings as Stage C:

- `contrast`;
- `auroc`;
- `validation_patch`.

Measure selected-channel fidelity for **every intervention size**

\[
k=1,2,\ldots,16.
\]

The full-patch (`k=16`) and no-op identities are integrity checks. The same-size random control is evaluated for every k using the frozen eight deterministic random permutations per model.

The validation replacement-energy-matched control is evaluated at the historically central budgets

\[
k\in\{1,2,4,8\},
\]

using eight closest same-size alternative sets per concept/ranking/budget. Matching diagnostics are retained; singleton settings are not assumed to be well matched.

## Main robustness estimand

Use `retention_1` as the reference profile. For profile `p`, block `b`, task/model setting `s`, ranking `r`, and budget `k`, define

\[
\Delta_{b,p,s,r}(k)
=F_{b,p,s,r}(k)-F_{b,\mathrm{retention1},s,r}(k).
\]

For compatibility with the prospective confirmation define the small-versus-large budget contrast

\[
B_{b,p,s,r}
=\frac{\Delta(4)+\Delta(8)}{2}
-\frac{\Delta(1)+\Delta(2)}{2}.
\]

The robustness study reports `B` for every non-reference profile, task, architecture and ranking. Within each task/architecture/ranking family, the five profile-versus-reference tests are Holm-adjusted. These are robustness inferences, not replacements for the single prospective primary test.

## Full budget curves

For each profile/task/architecture/ranking report the paired mean and 95% t interval of

- selected fidelity;
- release/profile minus `retention_1` selected fidelity;
- selected counterfactual accuracy;
- same-size-random relative usefulness.

At k in {1,2,4,8}, also report validation-energy-matched relative usefulness and matching error.

The complete k=1..16 curve is retained even if a simpler subset is highlighted in the manuscript.

## Schedule interpretation

The early/default/late release schedules form an ordered **timing sensitivity analysis**, not a randomized dose-response experiment. `template_init` and `retention_0p1` are boundary/reference profiles. We do not infer a monotone causal law from schedule order alone.

## Computational execution

Training may be parallelized across independent model jobs. Data for all planned block/task combinations must be generated before worker processes start, avoiding concurrent dataset creation. Every worker uses deterministic PyTorch algorithms and a fixed thread count.

The recommended workstation configuration is 8 workers with 2 PyTorch CPU threads each. Evaluation is performed with CUDA when available. Compute settings may change runtime but not the frozen statistical design; they are recorded in the execution manifest.

## Integrity

- Atomic per-model checkpoints and resume state are required.
- Completed models are never silently retrained under different source/config hashes in the same output root.
- Source hashes, Git HEAD, protocol commit/hash, Python/PyTorch versions, GPU identity and execution settings are recorded.
- Full-patch and no-op patch identities must pass existing numerical tolerances.
- All 1200 planned models remain in the analysis denominator unless a run is technically incomplete; technical failures are resumed/repaired rather than outcome-selected.
- No natural-image or human-interpretability claim follows from this robustness grid.

## Purpose

This stage answers a narrower question than general interpretability: whether the observed dependence of internal intervention fidelity on intervention budget is stable across fresh renderer blocks, task composition, shallow architecture depth, retention strength and release timing.