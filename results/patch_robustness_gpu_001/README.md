# GPU patching-robustness evaluation

## Provenance

- Source output: `outputs/patch_robustness/`
- Recorded command: `bash run_gpu.sh main`
- Imported on: 2026-09-08
- Device: CUDA, NVIDIA GeForce RTX 4060 Laptop GPU
- Batch size: 64
- Code changes during the run: none. The evaluator and protocol are tracked under `studies/cnn_patch_robustness/`; the imported model core is tracked under `studies/cnn_release_experiment/core.py`. Hashes are recorded in `design.json`.
- The separate smoke output was not merged into this main evaluation.

## Completion

Completed 80 of 80 planned checkpoint evaluations at epoch 200. This is a post hoc sensitivity analysis of existing CPU-trained models and test data, not an independent training replication. The imported files include per-evaluation JSON results, summary CSV tables, plot, report, and design provenance.

`import_manifest.sha256` contains hashes for every imported file. No file in this result directory exceeds 90 MiB at import time.
