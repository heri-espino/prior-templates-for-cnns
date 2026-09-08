# Prior templates for CNNs

Research on when template-initialized convolutional filters improve learning, and whether template alignment corresponds to independently measured concepts and causal usefulness.

**Current status:** the 400-run alignment study is complete. Stronger retention increases alignment, but all four primary causal-usefulness comparisons remain inconclusive after multiple-comparison correction. The retention-release follow-up is complete (200 runs × 200 epochs) and analyzed in the [updated paper draft](papers/retention_release_draft.md).

## Start here

| Study | Question | Status | Entry point |
|---|---|---|---|
| 01 — First pass | What does the original project do? | 28 runs; historical pilot with known limitations | [Report](studies/cnn_first_pass/REPORT.md) |
| 02 — Alignment and causal usefulness | Does stronger template retention improve independent concept/patching measurements? | 400 runs + 200 exploratory head evaluations; audited | [Report](studies/cnn_causal_milestone/REPORT.md) · [Short findings](studies/cnn_causal_milestone/FINDINGS.md) |
| 03 — Retention release | Do templates help early learning, then constrain adaptation? | 200 completed runs; exploratory analysis | [Paper draft](papers/retention_release_draft.md) · [Instructions](studies/cnn_release_experiment/README.md) |

The [learning-dynamics follow-up](studies/cnn_causal_milestone/LEARNING_DYNAMICS.md) examines all 16,000 saved epoch records from study 02. Template initialization has an early descriptive advantage in some settings, not all. Forty epochs did not establish a genuine plateau.

## Next experiment: GPU patching robustness

The next evaluation reuses saved final models and tests three validation-only channel rankings at four patch sizes. It requires CUDA by default, saves separate outputs and resumes per checkpoint.

```sh
bash run_gpu.sh check
bash run_gpu.sh smoke
bash run_gpu.sh main
```

See [GPU instructions](studies/cnn_patch_robustness/README.md) and [analysis plan](studies/cnn_patch_robustness/PROTOCOL.md). This is the robustness stage; new-task replication is not launched yet.

## Run the current experiment

Python 3.10–3.12 is recommended. From the repository root:

```sh
bash run.sh smoke
bash run.sh main
```

Windows PowerShell:

```powershell
.\run.ps1 smoke
.\run.ps1 main
```

The launchers install dependencies in the study's local environment. Main runs 200 models, each for 200 epochs. `pilot` runs one block (20 models). Run the same command after an interruption to resume. Avoid concurrent writers to one output folder.

**Already running the downloaded package? Keep that run in its current folder.** There is no need to stop, move or restart it. Its experiment/core source files are preserved byte-for-byte here. See [importing ongoing results](docs/IMPORT_RESULTS.md) when it finishes.

## Repository layout

```text
studies/
  cnn_first_pass/           # Study 01: original sources, repairs, pilot evidence
  cnn_causal_milestone/     # Study 02: frozen protocol/code and audited evidence
  cnn_release_experiment/   # Study 03: current runner, tests and reporting
.ai_handoff/               # Concise context, decisions, evidence and next actions
scripts/                   # Repository integrity checker
 docs/                     # Result import and evidence navigation
```

Historical study directories retain their names and internal organization deliberately: source hashes, audit scripts and report links depend on them. Navigation is organized here without rewriting the scientific record. Historical data, checkpoints and tables are versioned; new study-03 output and environments are ignored until a reviewed result import. Redundant distribution ZIPs are omitted except the small original input archive.

## Scientific interpretation

- Alignment measures kernel resemblance to an audited edge/corner/ring bank.
- Concept AUROC and localization use renderer annotations independent of that bank.
- Causal usefulness U measures selected-channel patching fidelity advantage over a random same-size channel set. It can reproduce an inaccurate model's behavior; always inspect accuracy too.
- Frozen-template TinyCNN accuracy on `single_shape` rises from 83.40% to 99.77% after changing only the classifier. This establishes a training/readout optimization gap at fixed features, not a universal representation limit.
- No publication novelty, general causal identifiability or universal negative effect is claimed.

For reproducibility, consult each study's README and protocol. Check imported file integrity with `python3 scripts/verify_import.py`. AI collaborators should read [AGENTS.md](AGENTS.md), then [.ai_handoff/START_HERE.md](.ai_handoff/START_HERE.md). No license is assigned yet; ownership/licensing of the original input needs review before choosing one.
