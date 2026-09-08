# Package validation

- A real two-epoch smoke run completed on CPU for two_concepts / TinyCNN / template_release, including checkpoint concept, localization, patching and head-refit diagnostics.
- Scheduling endpoints and paired downstream initialization passed automated checks.
- A simulated interruption after saving epoch 2 resumed to exactly the same model weights and non-timing histories as an uninterrupted three-epoch run.
- The Bash launcher passed syntax validation. Dependency installation and the Windows launcher were not exercised in a fresh environment.
- The full 200-model experiment has not been run. Smoke outputs are excluded from this source package.
