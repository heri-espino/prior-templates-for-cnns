# Validation

- Real epoch200 single_shape/TinyCNN/constant-retention checkpoint evaluated on CPU: original contrast/k4 U difference about −1.03e−8; full/no-op forwards passed.
- Tests compare new batched patching with the original evaluator for both architectures, cover no-op/full/sizes1,4,8, and undefined effects. Optional CUDA parity branch runs when CUDA exists.
- Identical rerun skips the completed smoke checkpoint (resume verified).
- Bash launcher syntax and historical import integrity checked.
- CUDA/RTX4060 execution not tested on development Mac. Run `bash run_gpu.sh check` then `smoke` on the owner's WSL machine. No GPU speed or numerical parity guarantee is inferred from CPU tests.
