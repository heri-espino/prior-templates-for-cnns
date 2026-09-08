# Random versus template priors in TinyCNN: reproduction and controlled first-pass analysis

**Status:** empirical first-pass report; not a claim of publication-ready evidence.  
**Source:** uploaded `cnn.zip`, recovered from the referenced conversation attachment. The requested `/mnt/data/cnn.zip` path was unavailable on this macOS host.  
**Terminology:** `random`, `template_init`, and `frozen_templates` are the repository regimes. `random_unitnorm` and `frozen_random_unitnorm` are added controls.

## Abstract

[Results inserted after completion and validation.]

## 1. Introduction and research questions

This project tests whether a small convolutional classifier can retain similarity to predefined visual templates while learning synthetic shape classification. We ask whether the repository's template priors change final accuracy, learning speed, out-of-distribution (OOD) behavior, kernel alignment and drift, class-conditional ablation, and pooled-vector patching. Added controls ask whether outcomes arise from template structure, normalization, or freezing.

The implementation contains initialization and freezing interventions, not a continuous prior-strength parameter, Gabor constraint, or regularization term. Its training loss is cross-entropy alone. This report therefore does not describe a measured λ frontier, semantic identifiability, or a new interpretability method. The earlier literature discussion motivates questions but is not treated as verified evidence or as a specification of what this archive implements. A systematic literature review and novelty assessment are outside this experimental report.

## 2. Source inspection and reproducibility

The archive contains 20 Python source files under `src/` and `scripts/`, alongside macOS metadata and Python bytecode. It contains no README, dependency lockfile, dataset, checkpoint, historical metric table, or archived experiment output. Consequently, we reproduce configurations from scripts; we cannot verify numerical agreement with earlier runs.

The original training entry point fails before training with `NameError: name 'model' is not defined`: its probe/reporting block was dedented out of `main()`. A second defect omits the mandatory `delta_cond` argument to `topk_patching`. The working copy repairs the indentation and supplies `cond_abl.delta_cond`. The original training file, captured failure, and unified patch are retained in `original/train.py`, `original_execution_failure.txt`, and `repairs.patch`. All other uploaded Python files remain unchanged; additions live alongside them. Synced project reference files were not modified.

The analysis keeps the repository's accuracy and probe definitions intact and labels corrected or additional measurements separately. Full commands, return codes, durations, configuration JSON, epoch logs, initial/final kernels, model checkpoints, raw probe arrays, environment versions, and the archive SHA-256 accompany this report.

## 3. Methods

### 3.1 Synthetic task and splits

`ShapesDataset` renders four classes: `line`, `circle`, `triangle`, `square`. Labels are sampled uniformly rather than stratified. Each 64×64 grayscale image contains one outlined shape on a background, inverted to make foreground values positive. The renderer draws at 4–8 times resolution and downsamples with Lanczos. It samples rotation, position, scale, line thickness, Gaussian pixel noise, and possible rectangular occlusion.

| Factor | Train / validation / ID test | OOD change |
|---|---|---|
| Rotation | 0 to π/4 | `ood_rot`: π/4 to 2π |
| Translation | x and y each −6 to +6 pixels | unchanged |
| Scale | 0.8 to 1.2 | unchanged |
| Thickness | 1 to 3 pixels | `ood_thick`: 4 to 6 |
| Noise SD | 0 to 0.10, then clipping to [0,1] | unchanged |
| Occlusion probability | 0.10 | `ood_occ`: 0.40 |
| Occluder width/height | each 6 to 18 pixels | unchanged |
| Supersampling | integer 4 to 8 | unchanged |

These are separately sampled splits, not matched counterfactual images. An occlusion draw can miss the foreground. Rotational symmetries mean `ood_rot` does not imply uniformly novel geometry for every class; circles are rotation-invariant.

For run seed s, the split RNG seeds are s+1 through s+6. Within a run, the regimes receive identical generated examples. Across consecutive run seeds, split RNG streams overlap: for example, seed-0 validation and seed-1 training share seed 2 and an identical image prefix. Similarly, seed-0 ID test and seed-1 validation share seed 3. This compromises independence of five-seed summaries and can place one run's evaluation images in another run's training set. Each model still trains only on its own training split; this is not evidence of within-run train/test leakage. Seed SDs and paired differences below are descriptive, not independent-replication confidence intervals.

### 3.2 Model and template bank

`TinyCNN` has one bias-free 1→16 convolution, 9×9 kernels and same padding, ReLU, spatial maximum pooling (`pooled presence vector` z), and a linear 16→4 classifier with bias. It has 1,364 parameters: 1,296 convolution weights and 68 classifier parameters. Under full freezing, only the 68 head parameters can update, although the convolution parameters retain `requires_grad=True` and a hook zeros their gradients.

The template bank has eight `edge_*`, four `corner_*`, and four `ring_*` entries. Each is mean-centered and L2-normalized. The oriented edges are first-derivative-like Gaussian filters, not Gabor kernels. `corner(theta)` normalizes the sum of two orthogonal oriented edges. Algebraically, because their isotropic Gaussian envelopes coincide, that sum is another oriented edge at theta+π/4. The report retains `corner` as the source name while auditing the actual construction. Ring radii are 1.5, 2.0, 2.5, and 3.0 in the 9×9 kernel.

| Condition | Initial convolution | Trainable convolution? | Provenance |
|---|---|---|---|
| `random` | PyTorch default random initialization | yes | repository baseline |
| `template_init` | all 16 template kernels | yes | repository |
| `frozen_templates` | all 16 template kernels | no updates | repository baseline |
| `random_unitnorm` | same random draw, centered and L2-normalized per kernel | yes | added normalization control |
| `frozen_random_unitnorm` | same centered, unit-norm random kernels | no updates | added freezing/structure control |

All conditions use the same head initialization within a seed: convolution and classifier are constructed before template replacement or added normalization. Neither replacement nor normalization consumes additional random numbers. Data creation uses local NumPy generators. The same seeded training shuffle stream is therefore used within a seed. Normalization controls match mean and norm, not the bank's rank, correlation, spatial spectrum, or orientation coverage.

### 3.3 Training and experiment scope

Adam uses learning rate 0.001, cross-entropy loss, batch size 128, no weight decay, no scheduler, and no early stopping. The final epoch is evaluated; validation maxima and time-to-threshold are descriptive and do not select a checkpoint. CPU computation uses four threads per training process. Independent experiment processes can contend for hardware, so their recorded wall times are not comparable performance benchmarks.

| Suite | Conditions | Seeds | Train / val / each test split | Epochs | Patching pairs |
|---|---|---|---|---|---|
| `v4_lowdata` | all 3 repository regimes | 0–4 | 200 / 400 / 600 | 100 | 1,200 |
| `controls` | 2 added controls | 0–4 | 200 / 400 / 600 | 100 | 1,200 |
| `v3_seed0` | all 3 repository regimes | 0 | 8,000 / 1,000 / 1,000 | 10 | 1,500 |

The v4 data sizes, seeds, epochs, model, and optimizer reproduce the existing low-data script configuration; probes are enabled instead of its default disabled setting. All available test items are used by the probes. The v3 configuration is reproduced for seed 0 only, not the full five-seed suite. The v2 eight-channel grid is not run: taking the first eight bank entries changes both width and template-family coverage, since it includes only `edge_*` filters. The full 16-channel comparison is the controlled focus here.

### 3.4 Repository metrics and their meanings

- **Accuracy:** `evaluate` computes the unweighted mean of batch accuracies. We retain `id_acc`, `ood_rot_acc`, `ood_thick_acc`, and `ood_occ_acc`, and independently recompute sample-weighted accuracy from checkpoints. With 400 validation examples, the last 16 examples receive 25% of the original validation metric's weight rather than 4%. With 600 test examples, the last 88 receive 20% rather than 14.7%. The supplemental majority baseline predicts the most common *training* class.
- **Alignment:** `alignment.mean_best_score` averages each mean-centered kernel's maximum signed cosine similarity over all templates. It permits repeated matches, is relative to this bank, and is not semantic selectivity or concept recovery. Sign matters with ReLU; the score does not maximize absolute cosine.
- **Drift:** mean-centered cosine with each kernel's own initialization and raw L2 displacement. High cosine means directional retention; zero L2 means no change. Mean centering in cosine can hide changes in DC response.
- **Class-conditional ablation:** with logits Vz+b, `delta_cond[i,c] = V[c,i] E[z_i | y=c]`. Specificity is the largest class contribution minus the mean of the others, averaged over channels. It is an exact head-level logit contribution, not an ablated-accuracy score or a calibrated semantic measure.
- **Single-channel patching:** replaces target pooled feature z_t[i] with source z_s[i] for differently labeled image pairs. The reported success is P(Δ source-versus-target margin > 0), not prediction transfer. Mean shift measures raw logit-margin effect size.
- **Top-k patching:** ranks channels by source-class `delta_cond`, patches k∈{1,2,4,8}, and uses the same positive-shift definition. Original ranking and evaluation both use test data. The key `success@maxk` actually stores the maximum success over all k, not necessarily the value at k=8. We preserve the field and report its meaning explicitly.
- **Learning speed:** existing validation `t98`, `t100`, and unnormalized trapezoidal AUC through epochs 10, 20, and 50. Failure to reach a threshold is censored at the training horizon, not a zero or a success. An AUC ending at epoch E integrates epochs 1…E, so its maximum is E−1. Shorter v3 runs do not support AUC through 20/50; those supplemental table cells are left undefined.

### 3.5 Additional controls and validation

Supplemental patching ranks channels using validation labels and activations, then evaluates on ID test pairs. At each k∈{1,2,4,8,16}, we compare against 20 fixed random class-specific channel rankings and report positive-margin shift, mean shift, prediction change, source-label prediction, and source-label prediction conditional on both original images being correctly classified. This last subset can differ between models; the unconditional comparison uses identical pairs within each seed.

No-op interventions must leave logits unchanged. Patching all 16 features must recover the source logits, since the head is linear; this is a sanity identity, not evidence of identified concepts. We check the ablation formula against direct head evaluation, reproduce original ID accuracy from every saved checkpoint, and verify frozen filters remain bitwise unchanged. Saved alignment matrices are also checked against direct float64 summation to a tolerance of 10⁻⁵. Some local NumPy matrix operations emitted numerical warnings; only finite, independently checked measurements are accepted. All patching intervenes on pooled z, not on spatial maps or rendered generative factors. Pairs are sampled with replacement from the same test bank; 1,200 pairs are not 1,200 independent images.

## 4. Results

[Tables and figures inserted from completed runs.]

## 5. Interpretation

[Interpretation inserted after results validation.]

## 6. Limitations

1. These are synthetic single-shape images, one shallow architecture, a small structured template bank, and a restricted optimization budget. Findings do not generalize automatically to natural images, deeper CNNs, or other priors.
2. Five adjacent seeds are dependent through the split-seed scheme. The v3 comparison has one seed. Descriptive SDs are not a substitute for independent replicated datasets and model seeds.
3. The template bank has redundant channels and `corner` entries that are edges in the implemented formula. Alignment with this bank does not establish semantic or causal identifiability.
4. Normalization-matched random controls leave rank, spectrum, redundancy, and directional coverage unmatched. They reduce confounding but do not isolate every feature of a template prior.
5. The original test-ranked top-k probe uses evaluation labels to choose channels. Validation-ranked supplements address this selection reuse, but the validation and test populations are still synthetic and the intervened feature combinations need not lie on the renderer's data manifold.
6. Positive margin movement can be arbitrarily small and need not change a prediction. Raw logit contribution and margin magnitudes are not directly calibrated across models; high specificity is not inherently superior interpretability.
7. The original accuracy weighting affects validation curves and threshold times. Supplemental final accuracy corrects weighting, but weighted historical validation curves cannot be reconstructed without epoch checkpoints, which the source does not save.
8. Max pooling discards spatial arrangement. Probes operate at the linear head and do not demonstrate recovery of rendering factors, localization, sufficiency of semantic parts, or full-network causal abstraction.
9. No historical outputs were supplied. Numerical historical reproduction cannot be claimed, and the execution repairs may not coincide with the author's earlier working version.
10. Final-epoch results do not establish converged performance or a complete accuracy–alignment frontier. No prior-strength sweep, confidence-calibrated hypothesis test, or confirmatory held-out experiment was performed.

## 7. Concrete next experiments

1. **Repair evaluation and seed isolation first.** Use `SeedSequence.spawn` or non-overlapping, named RNG streams for dataset, model, shuffle, and probes. Hold a disjoint final test bank fixed and pair treatments on it. Use sample counts in every accuracy average. Save epoch checkpoints or predictions. Repeat with at least 10 independent data/model blocks; report paired effect estimates and intervals at the independent-block level.
2. **Repair and factor the template bank.** Implement actual corner/junction patterns with a stated geometry, inspect rendered filters, and audit rank. Compare edges-only, rings-only, corrected corners, and mixed banks at matched channel count and norm. Add rank- and spectrum-matched random banks to determine whether observed differences arise from structure or redundant basis coverage.
3. **Separate initialization, freezing, and soft retention.** Keep the existing regimes as anchors; add a clearly defined normalized penalty λ‖W−T‖² (or another explicitly justified objective) across predeclared λ∈{0,10⁻⁴,10⁻³,10⁻²,10⁻¹,1}. Include zero-strength and frozen endpoints, tune only on validation, and evaluate final accuracy, weight alignment, and independent semantic/causal scores. This would be a new experiment, not a result already present here.
4. **Test optimization versus capacity.** Extend frozen-head training and sweep head learning rate with a validation-only selection rule. Compare validation loss trajectories and an independently optimized linear head on fixed pooled features. Extend trainable models to equal update budgets and compare low-data sizes 50, 200, 1,000, and 8,000. Distinguish compute budget from number of epochs.
5. **Measure semantic recovery using the renderer.** Retain complete metadata, including actual occluder placement and pixel masks; add matched images that change exactly one factor. Measure spatial activation overlap and selectivity for edges, junctions, or curvature rather than assigning whole-class semantics from kernel appearance.
6. **Strengthen causal probes.** Rank on training/validation only; use a disjoint pair bank; include no-op, random-channel, matched-magnitude, and sign controls. Report full prediction transfer and logit effects with uncertainty. Test factor-matched source/target pairs, account for nuisance changes, and compare pooled-vector with spatial-feature interventions.
7. **Target observed OOD failure modes.** Add class-specific rotation/thickness sweeps with identical base factors, measure actual visible occlusion fraction, and compare training augmentation against prior changes. This tests whether priors improve robustness beyond direct coverage of the shifted factor.

## 8. Conclusion

[Conclusion inserted after results validation.]

## Appendix A. Artifacts and reproduction

Run from the `cnn_first_pass` directory with the recorded Python environment:

```sh
.venv/bin/python run_experiments.py
.venv/bin/python run_controls.py
.venv/bin/python analyze.py
.venv/bin/python build_report.py
```

The launchers skip completed outputs; use a fresh output directory or archive existing `runs/` to retrain. `requirements-lock.txt` records installed dependencies. The source train entry point is also directly runnable using commands recorded in `execution.jsonl`.

- `PROTOCOL.md`: design specified before observed training results, including the control amendment.
- `archive_sha256.txt`, `source_manifest.json`, `repairs.patch`: provenance.
- `runs/`: per-run configurations, training logs, original metrics, probe arrays, kernels, and checkpoints.
- `analysis/per_run.csv`: all original and supplemental scalar metrics per run.
- `analysis/details.json`: split hashes/counts, confusion matrices, top-k arrays, supplemental intervention results.
- `analysis/audit.json`: numerical checks, environment, and template-bank audit.
- `analysis/summary.csv`: grouped mean and sample SD; threshold infinities are censoring, not numeric successes.
- `execution.jsonl`, `execution_progress.log`, `controls_progress.log`, `logs/`: complete execution evidence, including any failed attempts.

## Appendix B. Repository source references

Methods are grounded in `src/data/shapes.py`, `src/templates/primitives.py`, `src/models/cnn.py`, and the repaired `src/train.py`; metric definitions come from `src/interpret/alignment.py`, `drift.py`, `ablation.py`, and `patching.py`. Experiment presets are in `scripts/run_multiseed_v3.py`, `run_multiseed_v4_lowdata.py`, and `run_grid_v2.py`; learning-curve and threshold conventions come from `src/analysis/aggregate.py` and `scripts/aggregate_runs_v4_lowdata.py`. These supplied sources, rather than uncrosschecked historical chat claims, are the authoritative references for this report.
