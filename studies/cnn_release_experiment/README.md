# Do templates help early learning, then constrain later adaptation?

This standalone follow-up preserves the previous renderer, corrected edge/corner/ring bank, networks, independent concepts and matched patching metrics. It starts fresh dataset blocks at 2000. It does not overwrite the previous study. No main experiment has been run in this package.

## Start here

Install Python 3.10–3.12, extract this folder, and open a terminal inside it. The launcher creates its own environment and installs dependencies on its first invocation (internet required).

macOS / Linux:

```sh
bash run.sh smoke
bash run.sh main
```

Windows PowerShell:

```powershell
.\run.ps1 smoke
.\run.ps1 main
```

If PowerShell script execution is restricted, use the direct Python commands below under your normal local policy. The Windows launcher is provided but has not been tested on Windows.

`smoke` checks two epochs of one model, including real semantic and intervention evaluation. It is not scientific evidence. `main` runs **200 models: 2 tasks × 2 architectures × 10 blocks × 5 conditions, 200 epochs each**. Checkpoint diagnostics add substantial computation. This is a CPU implementation; no GPU is required. Actual runtime depends on your machine; use `pilot` (one block, 20 models) to estimate it. Pilot has its own folder but uses the same first block as main: do not count it as independent evidence.

To stop, press Ctrl+C. To resume, run the same command. Completed runs are skipped and partial runs resume from the last atomically saved epoch, including optimizer and shuffle state. Avoid two launchers writing to the same output directory simultaneously. Interrupted diagnostic evaluation is repeated safely. Settings/source changes require a new output directory.

For a smaller initial study:

```sh
bash run.sh main --blocks 3 --output outputs/three_blocks
```

To generate a report while training is stopped or still incomplete:

```sh
bash run.sh report outputs/main
```

The report is also generated automatically when training finishes.

## What is compared

| Condition | Initial first layer | Retention |
|---|---|---|
| random | Original random initialization | None |
| random_unitnorm | Centered, unit-norm random | None |
| template_init | Corrected templates | None |
| template_retention_1 | Corrected templates | λ=1 throughout |
| template_release | Corrected templates | λ=1 through epoch 10, linearly decreasing to zero at epoch 80 |

The release formula is `clip((80 - epoch) / 70, 0, 1)`. All first layers are trainable. Adam learning rate is 0.003, batch size 128; the loss is cross-entropy plus λ times squared distance to the initial template anchor divided by its squared norm. No learning-rate scheduler or early stopping is applied. Each epoch is four optimizer steps. Matched conditions share datasets, minibatch ordering and downstream initialization. Normalization matching addresses a confound observed in the earlier study.

The full run uses TinyCNN and TwoLayerCNN on single_shape and two_concepts. Each block has 512 training, 256 validation and 256 test images with matched renderer contexts and independent split streams. The default schedule is a hypothesis to test, not a demonstrated optimum. Training is extended to 200 epochs, but convergence is not assumed.

## Saved outputs

Under `outputs/main/`:

- `design.json`: settings and training/evaluation source hashes; created before training.
- `environment.json`: runtime versions and platform.
- `data/`: images, renderer masks, concepts and matched nuisance metadata.
- `runs/.../history.json`: every epoch's training/validation loss and accuracy, λ, alignment, kernel norm, gradient norm, validation pooled feature norm and training duration.
- `runs/.../latest.pt`: last complete epoch, model, optimizer, random states and complete history for resuming.
- `runs/.../epoch_0000.pt`: initial model. Additional checkpoints at epochs 1, 5, 10, 20, 40, 80, 120, 160 and 200 (plus the final epoch if customized).
- `runs/.../evaluations/epoch_*/`: test accuracy, independent concept AUROC, localization IoU, causal usefulness U, full patching results, probe arrays and fixed-feature classifier-refit diagnostics. Validation chooses units, thresholds and refit regularization. Test results do not change training. These evaluations use the same definitions as the previous milestone.
- `analysis/REPORT.md`, `learning_curves.png`, and CSV tables: automatically generated report, mean ± SD plots and raw tables.

Scheduled checkpoints contain the full model, so later kernel visualizations and additional analyses remain possible. Causal diagnostics are checkpoint-based, not every epoch. Initial checkpoint has no automatic causal evaluation. Training loss/accuracy are measured during updates; validation metrics use the completed epoch. Timing excludes diagnostic evaluation and is not a hardware-independent learning-speed measure.

## Interpretation plan

Compare paired independent blocks on early mean validation accuracy (epochs 1–10), optimizer steps to 95% validation accuracy with non-attainment reported, final accuracy, and the final 20-epoch validation-loss slope. Inspect accuracy together with alignment, renderer concepts and causal usefulness. Release and constant retention are identical until epoch 10 by construction; the test of releasing the constraint concerns later behavior.

Reports are descriptive, without automatic significance claims. Define a smallest meaningful effect and a multiple-comparison plan before confirmatory analysis. Repeated checkpoints are dependent; do not count them as independent samples. Do not use test curves to choose the release schedule or preferred epoch. A longer run showing lower performance still does not alone establish a representation ceiling. The saved refitted-head diagnostics help investigate optimization gaps.

## Direct Python setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_runner.py
.venv/bin/python experiment.py
.venv/bin/python summarize.py outputs/main
```

On Windows use `python` for environment creation and `.venv\Scripts\python.exe` thereafter. Dependencies have compatible ranges for portability; `environment.json` records the actual runtime. For exact environment archiving, save `python -m pip freeze` from the environment you use.

`core.py` is adapted from the previous frozen study into a standalone evaluation module. The original study files remain unchanged. `test_runner.py` tests scheduling, paired initialization and exact recovery from a simulated interruption; the smoke run exercises the real diagnostics.
