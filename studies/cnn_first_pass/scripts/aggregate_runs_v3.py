from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.analysis.aggregate import discover_runs, load_run, group_by_regime, aggregate_curve, aggregate_scalar_metric, aggregate_time_to_threshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs_v3")
    ap.add_argument("--outdir", type=str, default="reports_v3")
    ap.add_argument("--thr", type=float, default=0.98)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(runs_root)
    records = [load_run(d) for d in run_dirs]
    groups = group_by_regime(records)

    summary = {}
    rows = []
    for regime, recs in groups.items():
        curve = aggregate_curve(recs, key="val_acc")
        tthr = aggregate_time_to_threshold(recs, key="val_acc", thr=args.thr)
        id_acc = aggregate_scalar_metric(recs, "id_acc")
        ood_rot = aggregate_scalar_metric(recs, "ood_rot_acc")
        ood_thick = aggregate_scalar_metric(recs, "ood_thick_acc")
        ood_occ = aggregate_scalar_metric(recs, "ood_occ_acc")

        align_mean = [r.metrics["alignment"]["mean_best_score"] for r in recs]
        spec_mean = [r.metrics["conditional_ablation"]["mean_specificity"] for r in recs]

        summary[regime] = {
            "n": len(recs),
            "time_to_thr": tthr,
            "id_acc": id_acc,
            "ood_rot": ood_rot,
            "ood_thick": ood_thick,
            "ood_occ": ood_occ,
            "align_mean_best": {"mean": float(np.mean(align_mean)), "std": float(np.std(align_mean, ddof=1)) if len(align_mean)>1 else 0.0},
            "spec_mean": {"mean": float(np.mean(spec_mean)), "std": float(np.std(spec_mean, ddof=1)) if len(spec_mean)>1 else 0.0},
            "val_curve": {
                "epochs": curve["epochs"].tolist(),
                "mean": np.nan_to_num(curve["mean"]).tolist(),
                "std": np.nan_to_num(curve["std"]).tolist(),
            }
        }

        for r in recs:
            rows.append({
                "regime": regime,
                "seed": r.seed,
                "run_dir": str(r.run_dir),
                "id_acc": r.metrics["id_acc"],
                "ood_rot_acc": r.metrics["ood_rot_acc"],
                "ood_thick_acc": r.metrics["ood_thick_acc"],
                "ood_occ_acc": r.metrics["ood_occ_acc"],
                "align_mean_best": r.metrics["alignment"]["mean_best_score"],
                "spec_mean": r.metrics["conditional_ablation"]["mean_specificity"],
            })

    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    pd.DataFrame(rows).to_csv(outdir/"runs.csv", index=False)
    print(json.dumps({"runs": len(records), "regimes": list(groups.keys()), "outdir": str(outdir)}, indent=2))

if __name__ == "__main__":
    main()
