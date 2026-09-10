# Exhaustive robustness study

This directory contains the predeclared secondary robustness grid for the template-prior CNN paper.

It is deliberately separate from `studies/cnn_budget_confirmation/`, whose blocks 4000--4019 remain the single prospective confirmation sample.

## Design

The exhaustive stage uses blocks 5000--5049, both renderer tasks, both shallow architectures, and six template-prior profiles:

- `template_init`: lambda=0 throughout;
- `retention_0p1`: lambda=0.1 throughout;
- `retention_1`: lambda=1 throughout;
- `release_early`: linear 1 -> 0 over epochs 5--40;
- `release_default`: linear 1 -> 0 over epochs 10--80;
- `release_late`: linear 1 -> 0 over epochs 40--160.

This is 1200 models at 200 epochs. The final evaluation measures selected fidelity and random-control usefulness for every k=1..16, plus validation-energy-matched controls at k={1,2,4,8}.

## Windows 11: recommended execution

The native PowerShell launcher uses `scripts/use_conda_env.ps1`. It locates the user's Conda executable, creates the named environment `prior-templates-cnns` if needed, installs dependencies inside that user environment, verifies CUDA when requested, and invokes Python through `conda run`. Shell activation and administrator privileges are not required.

Run from the repository checkout:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\studies\cnn_exhaustive_robustness\run_exhaustive.ps1
```

The defaults are tuned for the RTX 4500 Ada / 24-core workstation:

```text
Device = cuda
Workers = 8
ThreadsPerWorker = 2
BatchSize = 256
CondaEnv = prior-templates-cnns
```

Override them with ordinary PowerShell parameters when needed:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\studies\cnn_exhaustive_robustness\run_exhaustive.ps1 `
  -Device cuda `
  -Workers 8 `
  -ThreadsPerWorker 2 `
  -BatchSize 256
```

The Conda helper searches the existing Windows user installation under `LOCALAPPDATA`, user-profile Miniconda/Anaconda installs, `CONDA_EXE`, and Conda already on `PATH`. No administrator install is attempted.

## Cross-platform Zsh alternative

```zsh
zsh studies/cnn_exhaustive_robustness/run_exhaustive.zsh
```

## Resume behavior

Training saves `latest.pt` atomically for every model. Re-running the same launcher skips completed models and resumes interrupted ones. The evaluator likewise skips completed checkpoint JSONs.

An existing output root is rejected if its frozen source/configuration manifest differs from the current run. To intentionally run modified code or settings, choose a new output root.

## Outputs

Default root: `results/exhaustive_robustness_001/`.

Important files after completion:

- `execution_manifest.json`: exact Git/protocol/source/environment provenance;
- `training/design.json`: frozen model grid;
- `evaluation/design.json`: frozen intervention evaluation;
- `analysis/REPORT.md`: compact robustness interpretation;
- `analysis/full_budget_curves.csv`: k=1..16 curves;
- `analysis/budget_contrasts_B.csv`: small-vs-large budget contrasts with Holm adjustment;
- `analysis/energy_matching_summary.csv`: matching quality;
- `analysis/selected_fidelity_budget_curves.png`: full contrast-ranking curves.

Do not inspect partial block outcomes to redesign the protocol. Technical interruptions should be resumed under the same output root and source hashes.
