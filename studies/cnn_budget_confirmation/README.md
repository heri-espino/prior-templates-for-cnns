# CNN intervention-budget prospective confirmation

This directory contains the pre-specified independent-block confirmation of the Stage C intervention-budget sensitivity result.

The inferential contract is frozen in [`PROTOCOL.md`](PROTOCOL.md). The planned data are renderer blocks 4000–4019, which are disjoint from the earlier stages. The primary quantity is selected-channel fidelity, not the baseline-relative score `U`.

## One-command execution

From the repository root:

```zsh
zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

The script performs, in order:

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

## Environment variables

The defaults are intended for a CUDA-capable local environment:

```zsh
PYTHON_BIN=python3 \
DEVICE=cuda \
BATCH_SIZE=64 \
THREADS=2 \
zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

To use an existing conda environment, activate it first and leave `PYTHON_BIN=python`, or point `PYTHON_BIN` at that environment's Python.

To install the study dependencies before execution:

```zsh
INSTALL_DEPS=1 zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

To evaluate on CPU instead of CUDA:

```zsh
DEVICE=cpu zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

Training deliberately reuses the original deterministic Stage B implementation and remains CPU-based; `DEVICE` controls the checkpoint patch evaluations. This avoids changing the training implementation solely for the confirmation study.

To use a different output directory:

```zsh
OUTPUT_ROOT="$PWD/results/budget_confirmation_local" \
zsh studies/cnn_budget_confirmation/run_confirmation.zsh
```

## Resume semantics

The workflow is safe to rerun with the same `OUTPUT_ROOT` under the same code/design state:

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
