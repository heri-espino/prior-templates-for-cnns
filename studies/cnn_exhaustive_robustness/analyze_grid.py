"""Analyze the predeclared 1200-model exhaustive robustness grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

PROFILES = (
    "template_init",
    "retention_0p1",
    "retention_1",
    "release_early",
    "release_default",
    "release_late",
)
METHODS = ("contrast", "auroc", "validation_patch")
KS = tuple(range(1, 17))
REFERENCE = "retention_1"


def interval(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    err = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - err, "ci_high": mean + err}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def load_results(root: Path):
    evaluation = root / "evaluation"
    files = sorted(evaluation.glob("runs/*/*/block*/*/epoch_0200.json"))
    rows, models, matching = [], [], []
    for p in files:
        v = json.loads(p.read_text())
        ident = {k: v[k] for k in ("task", "architecture", "block", "profile", "epoch")}
        rows.extend({**ident, **r} for r in v["rows"])
        models.append(
            {
                **ident,
                "test_acc": v["test_acc"],
                "alignment": v["alignment"],
                "full_patch_max_error": v["full_patch_max_error"],
                "noop_max_error": v["noop_max_error"],
                "checkpoint_sha256": v["checkpoint_sha256"],
            }
        )
        matching.extend({**ident, **m} for m in v["matching"])
    return pd.DataFrame(rows), pd.DataFrame(models), pd.DataFrame(matching), files


def verify(root, rows, models, files):
    design = json.loads((root / "training" / "design.json").read_text())
    expected_blocks = set(range(5000, 5050))
    assert design["start_block"] == 5000 and design["blocks"] == 50
    assert set(design["profiles"]) == set(PROFILES)
    assert set(rows.block) == expected_blocks
    assert set(rows.profile) == set(PROFILES)
    assert set(rows.method) == set(METHODS)
    assert set(rows.k) == set(KS)
    assert set(models.block) == expected_blocks
    expected_models = 50 * 2 * 2 * 6
    assert len(files) == expected_models, (len(files), expected_models)
    assert len(models) == expected_models
    assert not rows.duplicated(["task", "architecture", "block", "profile", "method", "k"]).any()
    assert float(models.full_patch_max_error.max()) <= 2e-5
    assert float(models.noop_max_error.max()) <= 2e-5
    return design


def summarize_curves(rows):
    rec = []
    for keys, s in rows.groupby(["task", "architecture", "profile", "method", "k"], sort=False):
        task, arch, profile, method, k = keys
        for metric in ("fidelity", "cf_accuracy", "causal_usefulness_random"):
            vals = s[metric].dropna().to_numpy(float)
            if len(vals):
                rec.append({"task": task, "architecture": arch, "profile": profile, "method": method, "k": int(k), "metric": metric, **interval(vals)})
        if int(k) in (1, 2, 4, 8):
            vals = s["causal_usefulness_energy"].dropna().to_numpy(float)
            if len(vals):
                rec.append({"task": task, "architecture": arch, "profile": profile, "method": method, "k": int(k), "metric": "causal_usefulness_energy", **interval(vals)})
    return pd.DataFrame(rec)


def paired_vs_reference(rows, metric="fidelity"):
    rec, per_block = [], []
    for task in sorted(rows.task.unique()):
        for arch in sorted(rows.architecture.unique()):
            for method in METHODS:
                for profile in PROFILES:
                    if profile == REFERENCE:
                        continue
                    for k in KS:
                        s = rows[(rows.task == task) & (rows.architecture == arch) & (rows.method == method) & (rows.k == k)]
                        a = s[s.profile == profile].set_index("block")[metric]
                        b = s[s.profile == REFERENCE].set_index("block")[metric]
                        d = (a - b).dropna().sort_index()
                        assert len(d) == 50
                        q = interval(d.values)
                        rec.append({"task": task, "architecture": arch, "profile": profile, "method": method, "k": k, "metric": metric, **q})
                        per_block.extend({"task": task, "architecture": arch, "profile": profile, "method": method, "k": k, "metric": metric, "block": int(block), "delta": float(value)} for block, value in d.items())
    return pd.DataFrame(rec), pd.DataFrame(per_block)


def budget_contrasts(per_block):
    rec, values = [], []
    for (task, arch, profile, method), s in per_block.groupby(["task", "architecture", "profile", "method"], sort=False):
        p = s.pivot(index="block", columns="k", values="delta").sort_index()
        assert len(p) == 50 and all(k in p for k in (1, 2, 4, 8))
        B = p[[4, 8]].mean(axis=1) - p[[1, 2]].mean(axis=1)
        q = interval(B.values)
        test = ttest_1samp(B.values, 0.0)
        rec.append({"task": task, "architecture": arch, "profile": profile, "method": method, "t_stat": float(test.statistic), "p_two_sided": float(test.pvalue), **q})
        values.extend({"task": task, "architecture": arch, "profile": profile, "method": method, "block": int(block), "B": float(value)} for block, value in B.items())
    table = pd.DataFrame(rec)
    table["p_holm_within_setting"] = np.nan
    for _, idx in table.groupby(["task", "architecture", "method"]).groups.items():
        idx = list(idx)
        table.loc[idx, "p_holm_within_setting"] = holm_adjust(table.loc[idx, "p_two_sided"].to_numpy(float))
    return table, pd.DataFrame(values)


def model_summary(models, training_root):
    # Add final validation accuracy from the immutable training history.
    val = []
    for _, r in models.iterrows():
        p = training_root / "runs" / r.task / r.architecture / f"block{int(r.block):04d}" / r.profile / "history.json"
        h = json.loads(p.read_text())
        val.append(float(h[-1]["val_acc"]))
    models = models.copy()
    models["val_acc"] = val
    rec = []
    for keys, s in models.groupby(["task", "architecture", "profile"], sort=False):
        task, arch, profile = keys
        for metric in ("val_acc", "test_acc", "alignment"):
            rec.append({"task": task, "architecture": arch, "profile": profile, "metric": metric, **interval(s[metric].to_numpy(float))})
    return models, pd.DataFrame(rec)


def matching_summary(matching):
    if matching.empty:
        return pd.DataFrame()
    rec = []
    for keys, s in matching.groupby(["task", "architecture", "profile", "method", "k"], sort=False):
        task, arch, profile, method, k = keys
        rec.append({
            "task": task,
            "architecture": arch,
            "profile": profile,
            "method": method,
            "k": int(k),
            "n_matches": len(s),
            "mean_relative_energy_error": float(s.relative_energy_error.mean()),
            "median_relative_energy_error": float(s.relative_energy_error.median()),
            "max_relative_energy_error": float(s.relative_energy_error.max()),
        })
    return pd.DataFrame(rec)


def make_curve_figure(curves, out):
    subset = curves[(curves.method == "contrast") & (curves.metric == "fidelity")]
    tasks = sorted(subset.task.unique())
    archs = sorted(subset.architecture.unique())
    fig, axes = plt.subplots(len(tasks), len(archs), figsize=(12, 8), squeeze=False)
    for i, task in enumerate(tasks):
        for j, arch in enumerate(archs):
            ax = axes[i, j]
            s = subset[(subset.task == task) & (subset.architecture == arch)]
            for profile in PROFILES:
                q = s[s.profile == profile].sort_values("k")
                ax.plot(q.k, q["mean"], marker="o", markersize=2.5, label=profile)
                ax.fill_between(q.k, q.ci_low, q.ci_high, alpha=0.08)
            ax.set_title(f"{task} / {arch}")
            ax.set_xlabel("patched channels k")
            ax.set_ylabel("selected fidelity")
            ax.set_xticks([1, 2, 4, 8, 12, 16])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Exhaustive intervention-budget curves: mean and 95% t intervals over 50 blocks")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(out / "selected_fidelity_budget_curves.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", default="results/exhaustive_robustness_001")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows, models, matching, files = load_results(root)
    design = verify(root, rows, models, files)
    curves = summarize_curves(rows)
    paired, paired_block = paired_vs_reference(rows, "fidelity")
    B, Bblock = budget_contrasts(paired_block)
    models, endpoints = model_summary(models, root / "training")
    match_summary = matching_summary(matching)

    curves.to_csv(out / "full_budget_curves.csv", index=False)
    paired.to_csv(out / "paired_fidelity_vs_retention1.csv", index=False)
    paired_block.to_csv(out / "paired_fidelity_vs_retention1_per_block.csv", index=False)
    B.to_csv(out / "budget_contrasts_B.csv", index=False)
    Bblock.to_csv(out / "budget_contrasts_B_per_block.csv", index=False)
    models.to_csv(out / "model_endpoints.csv", index=False)
    endpoints.to_csv(out / "model_endpoint_summary.csv", index=False)
    matching.to_csv(out / "energy_matching_rows.csv", index=False)
    match_summary.to_csv(out / "energy_matching_summary.csv", index=False)
    make_curve_figure(curves, out)

    contrast_B = B[B.method == "contrast"].copy()
    default_B = contrast_B[contrast_B.profile == "release_default"]
    lines = [
        "# Exhaustive template-prior robustness study",
        "",
        "Statistical status: secondary robustness study. It does **not** alter the frozen 20-block prospective confirmation.",
        "",
        f"Completed final-checkpoint evaluations: **{len(files)} / 1200**.",
        f"Fresh renderer blocks: **{design['start_block']}--{design['start_block'] + design['blocks'] - 1}**.",
        "",
        "## Default-release replication across settings",
        "",
        "The table reports the predeclared selected-fidelity budget contrast B for `release_default - retention_1` under the contrast ranking.",
        "",
        "| Task | Architecture | mean B | 95% interval | p | Holm p within setting |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in default_B.iterrows():
        lines.append(f"| {r.task} | {r.architecture} | {r['mean']:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {r.p_two_sided:.4g} | {r.p_holm_within_setting:.4g} |")

    lines += [
        "",
        "## Full profile map (contrast ranking)",
        "",
        "| Task | Architecture | Profile vs retention_1 | mean B | 95% interval | Holm p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in contrast_B.iterrows():
        lines.append(f"| {r.task} | {r.architecture} | {r.profile} | {r['mean']:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {r.p_holm_within_setting:.4g} |")

    lines += [
        "",
        "## Integrity and interpretation",
        "",
        f"Maximum full-patch numerical error: {models.full_patch_max_error.max():.3g}.",
        f"Maximum no-op numerical error: {models.noop_max_error.max():.3g}.",
        "",
        "The complete selected-fidelity curves for k=1..16 are in `full_budget_curves.csv` and `selected_fidelity_budget_curves.png`. Energy-matched diagnostics remain restricted to k={1,2,4,8} as frozen in the protocol.",
        "",
        "Release timing is interpreted as a sensitivity map, not as evidence for a monotone causal dose law. This renderer study does not establish human interpretability or natural-image generalization.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "models": len(files),
        "blocks": 50,
        "profiles": list(PROFILES),
        "budgets": list(KS),
        "maximum_full_patch_error": float(models.full_patch_max_error.max()),
        "maximum_noop_error": float(models.noop_max_error.max()),
    }, indent=2) + "\n")
    print("Report:", out / "REPORT.md")


if __name__ == "__main__":
    main()
