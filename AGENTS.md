# Working in this repository

Read `.ai_handoff/START_HERE.md` first. Read only the topic-specific handoff needed for the current request; do not ingest raw result tables or the import manifest for general context.

- Preserve studies 01 and 02 as historical evidence. Do not edit frozen protocols, source, data, checkpoints or reported outcomes. Create a new study/version for scientific changes.
- The owner is running study 03 externally. Do not move, overwrite, restart or assume access to that process. Its full outputs are not yet in this repository.
- Keep reported empirical outcomes separate from planned tests and exploratory analyses. All four study-02 primary tests are inconclusive after Holm correction.
- Never infer human interpretability from alignment alone. Keep actual metric names and accuracy alongside patching fidelity.
- Do not commit environments, credentials, temporary files or unreviewed active-run outputs. Do not load untrusted PyTorch checkpoint files; resume files use Python deserialization.
- Update the relevant short handoff when work changes. Prefer replacing stale status over appending repetitive conversation logs.
- Validate code changes with focused tests. Use `scripts/verify_import.py` for imported historical integrity; it intentionally fails if imported source files change, so version new work explicitly.
