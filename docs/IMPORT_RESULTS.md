# Import a running experiment safely

Keep the current process and output directory in place until it finishes. Do not run a second process against that folder. The downloaded study03 package and this repo share identical core/experiment sources at initial import.

When finished:

1. Preserve a backup of the entire output folder, including design.json, environment.json, data and runs.
2. Generate its report with the original package's `summarize.py`.
3. Copy the completed output into a new versioned result directory (for example `results/retention_release_001/`). Do not overwrite historical studies or another run.
4. Compare its design.json source hashes against the corresponding core.py and experiment.py. Count complete.json files against the design. Do not assume a partial run is complete.
5. Review checkpoint diagnostics, missing/undefined metrics, convergence flags and duplicated block IDs before analysis. Pilot and main share a block and are not independent replications.
6. Add an import manifest with hashes and a short provenance note, then commit reviewed outputs. If size becomes impractical, use an explicitly documented GitHub Release or LFS policy rather than silently omitting evidence.

Current study03 outputs are deliberately ignored under its working directory. A future results directory is not automatically ignored. Never commit a virtual environment or credentials.
