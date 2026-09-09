# Existing-checkpoint kernel analysis

Run **from the repository root**, in the Python environment used for your experiments:

```bash
git pull
bash run_kernel_analysis.sh
```

Windows PowerShell:

```powershell
git pull
.\run_kernel_analysis.ps1
```

If dependencies are missing, install once:

```bash
python -m pip install -r analysis/kernel_similarity/requirements.txt
```

No training, GPU, dataset loading or model forward passes are required. CPU execution is intentional for tiny weight matrices. Existing PyTorch GPU installations work without changes. All loads use `weights_only=True`; an unsupported checkpoint fails without falling back to arbitrary Python deserialization.

## Point to your local results

Default input is `results/retention_release_001`, containing `design.json` and `runs/`. If your complete experiment folder lives elsewhere:

```bash
bash run_kernel_analysis.sh --input /path/to/experiment/output --output /path/to/kernel-analysis
```

```powershell
.\run_kernel_analysis.ps1 -InputRoot "D:\experiments\release_run" -OutputRoot "D:\experiments\kernel_analysis"
```

The input must be the complete Stage B run: both tasks, both architectures, all five conditions and ten blocks, initial `.npy` weights, every scheduled `.pt` checkpoint, and evaluation `result.json` files. A metrics-only export is insufficient. Missing files fail explicitly; runs are never silently dropped. Source hash and saved design are checked. Outputs must be outside the input tree. Re-running replaces only derived outputs at the selected destination; use a new output directory to preserve an earlier analysis.

## Deliverables

- `REPORT.md`: all endpoint matching means ± sample SD.
- `matching_trajectories.png`: nearest versus one-to-one matching and gap over training.
- `kernels_*.png`: every condition at initialization and epoch 200, block 2000, plus original bank. Kernels are centered/unit-norm and independently reordered by assignment. Shared color scale; not raw amplitude.
- `matrices_*.png`: full 16×16 signed cosines with assignment marks, both endpoints, fixed block.
- `matching_vs_behavior.png`: existing accuracy, concept AUROC, localization, U and counterfactual accuracy versus assignment similarity, all 200 final models.
- `per_checkpoint.csv`, `per_channel.csv`, `summary.csv`: numerical outputs.
- `similarity_matrices.npz`: every 16×16 matrix. Matrix row n corresponds to row n in `per_checkpoint.csv`; no Python objects are stored.
- `within_condition_correlations.csv`: descriptive Spearman correlations within task/architecture/condition, ten blocks at epoch 200. Empty when undefined; no p-values or causal claim.
- `COMPLETE.json`: written last, with source/design/input hashes, versions, model count and alignment consistency check. Its absence means the run did not complete successfully; individual output files alone do not establish completeness.

Outputs default to `analysis/kernel_similarity/results/`. Upload that folder, or ZIP it, if you rerun on your laptop. This is a checkpoint morphology analysis, not human interpretability measurement or the deferred independent forward-evaluation audit. Read `PLAN.md` for the fixed scope and interpretation limits.

Metric checks:

```bash
python analysis/kernel_similarity/test_metrics.py
```
