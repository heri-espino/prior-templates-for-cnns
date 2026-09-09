# CNN patch-energy control

This directory implements the post-hoc validation-energy-matched control defined in `PROTOCOL.md`.

It **does not train models**. It reloads the existing Stage B checkpoints, reproduces the original Stage C rankings/patches, and adds same-size control channel sets matched on validation first-layer activation replacement energy.

## Run

From the repository root:

```bash
bash run_patch_energy_control.sh check
bash run_patch_energy_control.sh smoke
bash run_patch_energy_control.sh main
```

The default main output is `outputs/patch_energy_control/`. It is intentionally separate from imported evidence under `results/` and from the original Stage C output.

To regenerate summaries from a completed output tree:

```bash
bash run_patch_energy_control.sh report
```

## Expected outputs

`outputs/patch_energy_control/analysis/` contains:

- `metrics.csv`
- `paired_contrasts.csv`
- `matching_diagnostics.csv`
- `matching_quality.csv`
- `stage_c_consistency.csv`
- `two_concepts_key_contrasts.csv`
- `energy_matched_sensitivity.png`
- `REPORT.md`

## Scientific status

This control was designed after the original Stage C intervention-budget pattern was known. It is exploratory robustness analysis of the same trained models and test data. Even a successful matched-control result is not an independent confirmation.