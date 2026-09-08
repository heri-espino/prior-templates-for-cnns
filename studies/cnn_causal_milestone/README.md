# Alignment versus causal usefulness — milestone 2

Read [REPORT.md](REPORT.md) for the paper-structured methods, results, interpretation and next experiments. The [prospective protocol](PROTOCOL.md) defines the four primary comparisons and the distinction between alignment, semantic selectivity and intervention fidelity.

The study comprises **400 runs**: 2 synthetic tasks × 2 architectures × 10 independent blocks × 10 conditions. Models, images, renderer annotations, matched nuisance images, checkpoints, epoch logs, interventions and classifier-refit candidates are included. The first-pass project remains unchanged alongside this study.

## Reproduce the environment

Recorded environment: macOS arm64, Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2. Dependency versions are in `requirements-lock.txt`; other platforms may require compatible builds.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python test_foundations.py
.venv/bin/python run_study.py
.venv/bin/python analyze_study.py
.venv/bin/python nuisance_features.py
.venv/bin/python head_causal_diagnostic.py
.venv/bin/python audit_outputs.py
.venv/bin/python build_report.py
```

Run from this directory. Training uses two CPU threads per process. The default runner processes all blocks sequentially. Independent workers can be launched with `--worker 0 --workers 2` and `--worker 1 --workers 2`; each receives disjoint block IDs. The original execution used two such processes.

Existing `result.json` files are treated as completed runs and skipped. Move `runs/` aside before a full retraining. Existing `data/` is reusable and deterministic; move it aside to regenerate images. The supplementary nuisance analysis caches outputs in `analysis/nuisance_cache/`; clear that cache if changing models or the supplementary analysis. The source and protocol integrity audit intentionally fails if the frozen training code or protocol is edited; a new study should create a new version and freeze record rather than overwrite this one.

The report builder requires all 400 analyzed runs. It combines `REPORT.template.md`, generated tables/figures, and the reviewed empirical narratives in `analysis/abstract.txt`, `interpretation.txt`, `next.txt`, and `conclusion.txt`. Those narratives describe these results and must be reviewed if replacing the experiment. `--partial` analysis/report modes are for monitoring and are excluded from the final evidence package.

## Evidence map

- `design_freeze.json`: protocol and source hashes fixed before main outcomes.
- `foundation_audit.json`: corner geometry, rank, Fourier/Gram preservation, seed namespaces, matched initialization and retention-gradient checks.
- `data/<task>/blockXX/`: split arrays and context metadata; 20,480 main images plus matched nuisance variants.
- `runs/<task>/<architecture>/blockXX/<condition>/`: checkpoints and raw numerical evidence.
- `analysis/per_run.csv`: all main and clearly marked exploratory scalars.
- `analysis/primary_contrasts.csv`: the four prospective paired tests with Holm correction.
- `analysis/paired_contrasts.csv`: secondary comparisons, exploratory.
- `analysis/head_diagnostics.csv`: validation-selected alternative classifiers, with convergence diagnostics.
- `analysis/patching_all.json`: full/no-op/selected/random intervention outcomes and denominators.
- `analysis/nuisance_features.csv`: supplementary concept/nuisance first-layer response magnitudes.
- `analysis/head_causal_diagnostic.csv`: explicitly exploratory patches with the already selected alternative heads; no change to primary results.
- `audit.json`: independent checks on every completed dataset and checkpoint.
- `execution_worker*.jsonl`, `worker*.log`: execution records.

“Causal usefulness” is the specific score U defined in the protocol, not a claim of universally identified causal mechanisms. High kernel alignment, concept AUROC, foreground IoU and counterfactual fidelity are distinct outcomes.
