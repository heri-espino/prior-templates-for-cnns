# Retention-release experiment results

## Provenance

- Source output: `cnn_release_experiment/outputs/main/`
- Recorded command: `bash run.sh main`
- Imported on: 2026-09-08
- Code changes during the run: none. `core.py` and `experiment.py` were unchanged; their hashes are recorded in `design.json` and match `studies/cnn_release_experiment/`.
- `smoke` and `pilot` outputs were not imported. Pilot and main share a data block and are not independent experiments.

## Completion

The main run completed: 200 of 200 planned runs, covering 2 tasks × 2 architectures × 10 blocks × 5 conditions, with 200 epochs per run. The imported tree includes the generated datasets, checkpoints, histories, evaluations, report, plots, and CSV tables.

The report is descriptive. Interpret paired blocks, retain the actual metric names and accuracy alongside alignment and patching fidelity, and do not infer human interpretability from alignment alone.

`import_manifest.sha256` contains hashes for every imported file. No file in this result directory exceeds 90 MiB at import time.
