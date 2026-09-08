# Context in one minute

Owner: heri-espino. Objective: test when template priors help learning and when alignment corresponds to meaningful, causally useful features.

1. Study 01: 28-run pilot, historical flaws documented. Do not use its numbers as clean confirmatory evidence.
2. Study 02: 400 completed runs across 2 tasks × 2 architectures × 10 blocks × 10 conditions. Independent streams, sample-weighted metrics and corrected bank. Final audit passed. Strong alignment manipulation; all four primary U tests inconclusive after Holm correction.
3. Study 03: complete and imported at results/retention_release_001 (200 models × 200 epochs). All 12,129 manifest hashes verified. New exploratory analysis checks 40,000 epoch rows and 1,800 checkpoint evaluations against raw JSON; paper: papers/retention_release_draft.md. Template_init catches normalized random at99.30% on two_concepts/TinyCNN. Release improves compositional accuracy over constant retention, but ΔU is +.061 shallow and −.058 deeper. No universal early benefit or proven plateau.

Read next only as needed:
- `RESULTS.md`: key numerical findings and interpretation guardrails.
- `OPERATIONS.md`: commands, resume behavior and where evidence lives.
- `NEXT.md`: immediate next work after owner returns results.
- `DECISIONS.md`: design choices and reasons.

The top-level README links the reports. `import_manifest.json` is machine-readable provenance, not reading material; `python3 scripts/verify_import.py` checks it. Source folders outside this checkout were copied, not moved. No remote experiment monitoring is configured.

GPU robustness is complete and reviewed: all80 evaluations uploaded. See analysis/patch_robustness_gpu_001/REPORT.md. Budget/ranking dependence prevents broadly confirming the original architecture claim; new-task replication deferred. Reproduce review: bash review_robustness.sh.

Canonical manuscript: papers/main.md (one paper across stages A/B/C), with full supplementary tables and corpus-grounded literature_comparison.md. Old retention_release_draft.md redirects. Final independent reproducibility review deferred by owner; do not claim it completed. Literature added at ca45a5e,39 parsed papers with PDFs.
