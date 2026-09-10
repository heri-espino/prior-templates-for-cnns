# CNN intervention-budget prospective confirmation

This directory contains the pre-specified independent-block confirmation of the Stage C intervention-budget sensitivity result.

The inferential contract is frozen in [`PROTOCOL.md`](PROTOCOL.md). The planned data are renderer blocks 4000–4019, which are disjoint from the earlier stages. The primary quantity is selected-channel fidelity, not the baseline-relative score `U`.

## Windows 11: recommended one-command execution

The Windows launcher uses the user's Miniconda installation, creates the project environment if needed, and runs through `conda run`. It does not require administrator privileges or `conda activate`.

From PowerShell:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\studies\cnn_budget_confirmation\run_confirmation.ps1
```

The repository root also contains `run_paper_experiments.cmd`, which is the simplest Windows entry point when running both the confirmation and exhaustive stages.

## Cross-platform Zsh alternative

```zsh
zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

Both launchers perform, in order:

1. deterministic Stage-B-style training for 80 fresh models (`two_concepts`, two architectures, retention/release, 20 blocks);
2. standard patch robustness evaluation for all three rankings and `k={1,2,4,8}`;
3. validation replacement-energy-matched control evaluation;
4. the frozen prospective analysis and primary decision rule.

Default output:

```text
results/budget_confirmation_001/
├── execution_manifest.json
├── training/
├── patch_eval/
├── energy_eval/
└── analysis/
    └── budget_confirmation/
        ├── REPORT.md
        ├── decision.json
        ├── primary_curve.csv
        ├── treatment_contrasts.csv
        ├── per_block_treatment_effects.csv
        ├── budget_contrasts.csv
        ├── budget_contrasts_per_block.csv
        ├── integrity_checks.csv
        ├── energy_U_contrasts.csv
        ├── energy_selected_fidelity_check.csv
        └── energy_matching_diagnostics.csv
```

`execution_manifest.json` records the protocol commit/hash, repository head at launch, source SHA-256 hashes, environment, GPU, and exact frozen design.

## Windows configuration overrides

PowerShell parameters can be supplied directly, for example:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\studies\cnn_budget_confirmation\run_confirmation.ps1 `
  -Device cuda `
  -BatchSize 64 `
  -Threads 2 `
  -CondaEnv prior-templates-cnns
```

The Conda helper searches `LOCALAPPDATA\miniconda3`, `LOCALAPPDATA\anaconda3`, user-profile installs, `CONDA_EXE`, and Conda already on `PATH`.

Training deliberately reuses the original deterministic Stage B implementation and remains CPU-based; `Device` controls the checkpoint patch evaluations. This avoids changing the training implementation solely for the confirmation study.

## Resume semantics

The workflow is safe to rerun with the same output root under the same code/design state:

- training resumes from atomic epoch checkpoints;
- completed model runs are skipped;
- completed patch evaluations are skipped;
- completed energy-matched evaluations are skipped;
- the final analysis is regenerated deterministically.

If an immutable design or relevant source hash differs from the existing `execution_manifest.json`, the launcher stops and requires a new output directory rather than mixing evidence from different implementations.

## Primary decision

For TinyCNN with the `contrast` ranking, per block

\[
\Delta_b(k)=F^{\mathrm{release}}_{\mathrm{selected},b}(k)-F^{\mathrm{retention}}_{\mathrm{selected},b}(k),
\]

and

\[
B_b=\frac{\Delta_b(4)+\Delta_b(8)}{2}-\frac{\Delta_b(1)+\Delta_b(2)}{2}.
\]

The result is mechanically marked `CONFIRMED` only when all 20 planned blocks exist, mean `B>0`, the two-sided 95% Student-t interval has lower bound above zero, and the two-sided one-sample t-test has `p<0.05`.

No other architecture, ranking, intervention size, random-control score, or energy-matched score can change that primary decision.
