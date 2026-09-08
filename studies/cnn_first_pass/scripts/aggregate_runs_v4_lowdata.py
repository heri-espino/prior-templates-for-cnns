from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.analysis.aggregate import discover_runs, load_run, group_by_regime, aggregate_curve, aggregate_time_to_threshold
from src.analysis.kernels import mean_kernels_across_runs
from src.templates.primitives import TemplateSpec

def auc_curve(y: np.ndarray, x: np.ndarray) -> float:
    # trapezoidal, ignoring NaNs
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(y[m], x[m]))

def first_epoch_ge(df: pd.DataFrame, key: str, thr: float) -> float:
    df = df.sort_values("epoch")
    hit = df[df[key] >= thr]
    return float(hit.iloc[0]["epoch"]) if len(hit) else float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs_v4_lowdata")
    ap.add_argument("--outdir", type=str, default="reports_v4_lowdata")
    ap.add_argument("--thr_fast", type=float, default=0.98)
    ap.add_argument("--thr_final", type=float, default=1.0)
    ap.add_argument("--auc_epochs", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--canonicalize_kernels", type=int, default=1)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(runs_root)
    records = [load_run(d) for d in run_dirs]
    groups = group_by_regime(records)

    # per-run table
    rows = []
    for r in records:
        df = r.train_log.sort_values("epoch")
        t_fast = first_epoch_ge(df, "val_acc", args.thr_fast)
        t_final = first_epoch_ge(df, "val_acc", args.thr_final)
        row = {
            "regime": r.regime,
            "seed": r.seed,
            "run_dir": str(r.run_dir),
            "val_max": float(df["val_acc"].max()),
            f"t{int(args.thr_fast*100)}": t_fast,
            f"t{int(args.thr_final*100)}": t_final if np.isfinite(t_final) else float("inf"),
            "delta_t_final_minus_fast": (t_final - t_fast) if (np.isfinite(t_final) and np.isfinite(t_fast)) else float("inf"),
            "id_acc": float(r.metrics.get("id_acc", float("nan"))),
            "ood_rot_acc": float(r.metrics.get("ood_rot_acc", float("nan"))),
            "ood_thick_acc": float(r.metrics.get("ood_thick_acc", float("nan"))),
            "ood_occ_acc": float(r.metrics.get("ood_occ_acc", float("nan"))),
            "align_mean_best": float(r.metrics.get("alignment", {}).get("mean_best_score", float("nan"))),
            "run_probes": int(r.metrics.get("run_probes", 1)),
        }
        x = df["epoch"].to_numpy(dtype=np.float32)
        y = df["val_acc"].to_numpy(dtype=np.float32)
        for E in args.auc_epochs:
            m = x <= E
            row[f"auc_{E}"] = auc_curve(y[m], x[m])
        rows.append(row)

    runs_df = pd.DataFrame(rows).sort_values(["regime","seed"])
    runs_df.to_csv(outdir/"runs_table.csv", index=False)

    # aggregated summary per regime
    summary = {}
    for reg, recs in groups.items():
        curve = aggregate_curve(recs, key="val_acc")
        t_fast = aggregate_time_to_threshold(recs, key="val_acc", thr=args.thr_fast)
        t_final = aggregate_time_to_threshold(recs, key="val_acc", thr=args.thr_final)

        # kernel means (final)
        Ws = [rr.conv_W_final for rr in recs if rr.conv_W_final.size != 0]
        bts = [np.load(rr.run_dir/"alignment_best_template.npy") for rr in recs if (rr.run_dir/"alignment_best_template.npy").exists()]
        bss = [np.load(rr.run_dir/"alignment_best_score.npy") for rr in recs if (rr.run_dir/"alignment_best_score.npy").exists()]
        km = None
        if len(Ws) > 0:
            km = mean_kernels_across_runs(
                Ws,
                best_templates=bts if len(bts)==len(Ws) else None,
                best_scores=bss if len(bss)==len(Ws) else None,
                template_spec=TemplateSpec(),
                do_canonicalize=(args.canonicalize_kernels==1),
            )
            np.save(outdir/f"{reg}_kernels_mean.npy", km.mean)
            np.save(outdir/f"{reg}_kernels_std.npy", km.std)

        # AUCs from runs_df
        sub = runs_df[runs_df["regime"]==reg]
        auc_stats = {}
        for E in args.auc_epochs:
            col = f"auc_{E}"
            vals = sub[col].to_numpy(dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            auc_stats[col] = {
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std(ddof=1)) if vals.size>1 else 0.0,
                "n": int(vals.size),
            }

        summary[reg] = {
            "n": len(recs),
            "val_curve": {
                "epochs": curve["epochs"].tolist(),
                "mean": np.nan_to_num(curve["mean"]).tolist(),
                "std": np.nan_to_num(curve["std"]).tolist(),
            },
            "time_to_fast": t_fast,
            "time_to_final": t_final,
            "auc": auc_stats,
            "kernel_mean_saved": bool(km is not None),
            "kernel_order_note": (km.order_note if km is not None else ""),
        }

    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"runs": len(records), "regimes": list(groups.keys()), "outdir": str(outdir)}, indent=2))

if __name__ == "__main__":
    main()
