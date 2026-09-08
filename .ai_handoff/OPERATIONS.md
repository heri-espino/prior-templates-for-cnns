# Operations and evidence map

Root launchers delegate to studies/cnn_release_experiment. `bash run.sh smoke`, `pilot`, `main`, or `report outputs/main`. Paths passed to the delegated runner resolve relative to that study directory. Windows wrapper provided, not platform-tested.

Main defaults: CPU,2 threads; 10 blocks beginning2000; single_shape/two_concepts; TinyCNN/TwoLayerCNN; random/random_unitnorm/template_init/template_retention_1/template_release; Adam.003; batch128; 200 epochs. λ release holds1 through10 then reaches0 at80.

Every epoch atomically saves latest.pt including optimizer/shuffle state and history. Scheduled checkpoints1,5,10,20,40,80,120,160,200 plus initial/final. Scheduled evaluations save semantic AUROC, localization, U, patching, refit diagnostics. Identical settings/source required for resume. Outputs ignored in Git. Do not run simultaneous writers. Core/experiment copied unchanged from supplied package; completed owner outputs are imported under results/retention_release_001.

Tests: smoke real diagnostics passed; test_runner.py verified schedule, matched downstream initialization, exact simulated-interruption recovery. Bash syntax passed. No full study03 execution here; no Windows/fresh installer validation.

Historical study folders remain siblings because study02/audit_outputs.py references ../cnn_first_pass. Historical environments excluded. Import manifest records every copied source/evidence file; scripts/verify_import.py validates hashes and counts. Scientific reproduction commands live in each study README.

2026-09-08: study03 completed. Run analysis/retention_release_001/analyze.py with study dependencies to regenerate derived analysis. Final report: papers/retention_release_draft.md. No imported evidence is edited.
