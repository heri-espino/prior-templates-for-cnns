# GPU robustness evaluation — run this next

In the repository's WSL terminal:

```sh
git pull --ff-only
bash run_gpu.sh check
bash run_gpu.sh smoke
bash run_gpu.sh main
```

This is the first, evaluation-only part of the agreed final experiments. It reuses 80 saved final checkpoints (release/constant retention, both tasks/architectures, ten blocks). No retraining. It tests three validation-only rankings × four intervention sizes. Read [PROTOCOL.md](PROTOCOL.md) for exact definitions and exploratory status. The new annotated-task replication remains conditional on these results.

The launcher prefers the downloaded package's existing `cnn_release_experiment/.venv/bin/python`, then the study environment, then python3. Override explicitly if needed:

```sh
PYTHON=/path/to/your/.venv/bin/python bash run_gpu.sh check
```

Your prior environment already reported a CUDA-enabled PyTorch build; the GPU check verifies driver access and executes a real CUDA operation. If CUDA is unavailable, the runner stops. It never silently falls back to CPU. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for a compatible CUDA installation if needed. In WSL, also check `nvidia-smi`. The launcher does not overwrite your existing environment.

All inference/patching runs on GPU, float32, without mixed precision or TF32. Validation ranking/statistical summaries use CPU. Batch64 is a conservative starting point for the RTX4060 laptop; only one small CNN and its validation/test activation maps are retained at a time. If memory is constrained:

```sh
bash run_gpu.sh main --batch-size 32 --output outputs/patch_robustness_batch32
```

Input defaults to `results/retention_release_001`. Output is separate at `outputs/patch_robustness`. Stop with Ctrl+C; repeat the identical command to resume. Completed checkpoints are skipped. Device, batch size, source or input changes require a new output folder. Do not launch two writers to one folder.

To evaluate additional *existing* checkpoints, explicitly choose another output:

```sh
bash run_gpu.sh main --epochs 40 80 200 --output outputs/patch_robustness_trajectory
```

Default main is 80 evaluations; trajectory option is240. Smoke evaluates one checkpoint and saves separately; it is not independent evidence.

Outputs: per-checkpoint JSON including validation rankings and controls; `analysis/metrics.csv`; paired exploratory intervals; `analysis/legacy_comparison.csv`; `analysis/robustness.png`; `analysis/REPORT.md`. Original-k4 GPU results are compared against stored CPU outcomes so numerical discrepancies remain visible. No original files or checkpoints are modified. Checkpoints use restricted `weights_only=True` loading.

CPU tests and a real-checkpoint CPU smoke run are performed during development. An RTX4060/CUDA device is unavailable on the development Mac, so **GPU execution itself must be checked by your smoke run**. This is not a claim of GPU hardware validation. CPU mode is available explicitly for testing (`--device cpu`), not used by default.
