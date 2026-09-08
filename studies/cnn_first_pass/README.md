# CNN first-pass experiment package

Start with [the paper-structured report](REPORT.md). It contains methods, measured results, interpretation, limitations, figures, and prioritized next experiments.

- **28 completed runs:** 15 original low-data runs, 10 added normalization/frozen controls, 3 larger-data seed-0 runs.
- **Results:** [per-run CSV](analysis/per_run.csv), [full diagnostics](analysis/details.json), [numerical audit](analysis/audit.json), [design checks](analysis/design_checks.json).
- **Original code:** `cnn_original.zip` is the uploaded archive; `src/` is the working source with only the two documented training repairs. See `repairs.patch` and `source_manifest.json`.
- **Saved experiments:** `runs/` contains configurations, checkpoints, learning curves, kernel images, original metrics and raw probe arrays. Logs include failed development attempts as well as successful runs.

## Reproduce

The recorded environment is macOS arm64, Python 3.9, PyTorch 2.8.0 and NumPy 2.0.2. Installed package versions are in `requirements-lock.txt`; other platforms may need compatible package versions.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python run_experiments.py
.venv/bin/python run_controls.py
.venv/bin/python verify_design.py
.venv/bin/python analyze.py
.venv/bin/python build_report.py
```

Training launchers skip completed outputs. To retrain, move the included `runs/` directory aside first. Supplemental analysis caches are checkpoint-hash keyed; move `analysis/cache/` aside to force recomputation after changing analysis code. `build_report.py` requires all 28 analyzed runs and rebuilds the paper from `REPORT.template.md` plus the empirically reviewed text in `analysis/abstract.txt`, `interpretation.txt`, and `conclusion.txt`. Those narratives describe this experiment and must be reviewed if new runs replace it.

To reproduce the original aggregate tables:

```sh
.venv/bin/python -m scripts.aggregate_runs_v4_lowdata --runs_root runs/v4_lowdata --outdir analysis/repository_v4
.venv/bin/python -m scripts.aggregate_runs_v3 --runs_root runs/v3_seed0 --outdir analysis/repository_v3
```

The original v4 aggregation also saves canonicalized mean kernels. Those are descriptive visual summaries, not matched functional channels: sorting by template assignment is not a bijective channel matching, and sign-flipping a ReLU kernel is not function-preserving. The report instead shows individual-run kernels in their original order.

The supplied archive had no historical results to compare against. “Reproduction” here means execution of its script configurations after minimal documented repairs. No full v2 grid or five-seed v3 suite was run.
