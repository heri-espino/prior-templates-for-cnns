# First-pass protocol (specified before observing trained results)

1. Preserve uploaded Python sources; repair only the displaced `main()` block and missing `delta_cond` argument in `src/train.py`. Record a diff.
2. Attempt the existing v4 low-data experiment: all three repository regimes, five seeds 0–4, 200/400/600 train/validation/test items, 100 epochs, 16 channels, 9×9 kernels, 64×64 images, Adam 0.001, batch 128. Enable existing ablation and patching probes (1200 mismatched pairs; all 600 test items) for analysis. Final epoch is the evaluation checkpoint.
3. Reproduce the v3 configuration on seed 0 (8000/1000/1000; 10 epochs; 1500 pairs) for all three regimes, subject to measured runtime. This is a single-seed configuration reproduction, not reproduction of the full five-seed grid.
4. Keep repository batch-averaged accuracy and probe definitions intact. Supplement with sample-weighted accuracy, per-class confusion, majority-class baseline, validation-ranked top-k patches, no-op and random-channel patch controls, and actual prediction changes. Use identical examples and pair draws within each seed across regimes.
5. Audit template rank, redundancy, normalization, frozen-weight invariance, data split identities and metric semantics. Distinguish weight alignment from semantic or causal identification.
6. Report seed-level measurements and mean ± sample SD, with paired differences as descriptive values. No significance claims: the repository's adjacent split seeds induce dependencies across runs.
7. If further controls are feasible, compare scale-matched random initialization and frozen random filters using separated seed blocks; mark these as added controls rather than repository regimes.

## Control design amendment before control training

Add `random_unitnorm` and `frozen_random_unitnorm` at the v4 settings and seeds 0–4. Center and L2-normalize each initial random kernel to match the template normalization, retaining the paired classifier initialization and minibatch stream. The frozen version zeros convolution gradients, as the repository does. These two added controls separate kernel normalization and trainability from template structure. Adjacent-seed dependence remains disclosed; separated seeds are reserved for a confirmatory experiment.
