"""Evaluate the exhaustive robustness checkpoints across all intervention budgets.

Selected fidelity and same-size random controls are computed for k=1..16.
Validation-energy-matched controls are additionally computed at k={1,2,4,8}.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "studies" / "cnn_release_experiment"))
import core  # noqa: E402

KS = tuple(range(1, 17))
MATCH_KS = (1, 2, 4, 8)
METHODS = ("contrast", "auroc", "validation_patch")
N_CONTROLS = 8
EPS = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


@torch.inference_mode()
def collect(model, x, batch, device):
    hs, ls = [], []
    for st in range(0, len(x), batch):
        h = model.features(torch.from_numpy(x[st : st + batch]).to(device))
        hs.append(h)
        ls.append(model.tail(h))
    return torch.cat(hs), torch.cat(ls)


@torch.inference_mode()
def patch(model, H, orders, k, pair, batch):
    b, c, g = pair
    orders = torch.as_tensor(orders, device=H.device)
    out = []
    for st in range(0, len(b), batch):
        bs, cs, gs = b[st : st + batch], c[st : st + batch], g[st : st + batch]
        h = H[bs].clone()
        ix = orders[gs, :k]
        bi = torch.arange(len(bs), device=H.device)[:, None]
        h[bi, ix] = H[cs][bi, ix]
        out.append(model.tail(h).softmax(1))
    return torch.cat(out)


def pair_tensors(task, n, device):
    return tuple(torch.as_tensor(x, device=device) for x in core.pairs(task, n // 4))


def measure(prob, p0, p1, y, good):
    den = float((p1 - p0).square().sum())
    pred = prob.argmax(1)
    return {
        "fidelity": 1 - float((prob - p1).square().sum()) / den if den >= 1e-10 else None,
        "cf_accuracy": float((pred == y).float().mean()),
        "agreement": float((pred == p1.argmax(1)).float().mean()),
        "cf_accuracy_both_correct": float((pred[good] == y[good]).float().mean()) if good.any() else None,
    }


def rankings_from_validation(model, H, L, task, concepts, batch, device):
    z = H.amax((2, 3)).cpu()
    contrast, _, _ = core.concept_orders(z, concepts)
    nc = contrast.shape[0]

    auc_scores = np.array(
        [[abs(core.auc(concepts[:, c], z[:, ch].numpy()) - 0.5) for ch in range(16)] for c in range(nc)]
    )
    auc_order = np.argsort(-auc_scores, axis=1, kind="stable")

    pair = pair_tensors(task, len(H), device)
    b, c, g = pair
    p0, p1 = L[b].softmax(1), L[c].softmax(1)
    scores = np.zeros((nc, 16), dtype=np.float64)
    valid = []
    for concept in range(nc):
        valid.append(float((p1[g == concept] - p0[g == concept]).square().sum()) >= 1e-10)
    for ch in range(16):
        o = np.tile(np.r_[ch, np.delete(np.arange(16), ch)], (nc, 1))
        prob = patch(model, H, o, 1, pair, batch)
        for concept in range(nc):
            mask = g == concept
            den = float((p1[mask] - p0[mask]).square().sum())
            scores[concept, ch] = (
                1 - float((prob[mask] - p1[mask]).square().sum()) / den if valid[concept] else 0.0
            )
    return {
        "contrast": contrast,
        "auroc": auc_order,
        "validation_patch": np.argsort(-scores, axis=1, kind="stable"),
    }, scores, valid


def replacement_energy(H, pair, n_concepts):
    b, c, g = pair
    out = np.zeros((n_concepts, H.shape[1]), dtype=np.float64)
    for concept in range(n_concepts):
        d = H[c[g == concept]] - H[b[g == concept]]
        out[concept] = d.square().sum((0, 2, 3)).detach().cpu().numpy().astype(np.float64)
    return out


def subset_tables():
    tables = {}
    for k in MATCH_KS:
        combos = np.array(list(itertools.combinations(range(16), k)), dtype=np.int64)
        mask = np.zeros((len(combos), 16), dtype=np.float64)
        mask[np.arange(len(combos))[:, None], combos] = 1.0
        tables[k] = (combos, mask)
    return tables


SUBSETS = subset_tables()


def energy_matched_orders(ranking, energy, k):
    combos, combo_mask = SUBSETS[k]
    nc = energy.shape[0]
    controls = [np.empty((nc, 16), dtype=np.int64) for _ in range(N_CONTROLS)]
    diagnostics = []

    for concept in range(nc):
        selected = np.sort(ranking[concept, :k]).astype(np.int64)
        selected_mask = np.zeros(16, dtype=np.float64)
        selected_mask[selected] = 1.0
        target = float(selected_mask @ energy[concept])
        values = combo_mask @ energy[concept]
        rel = np.abs(values - target) / (target + EPS)
        same = np.all(combos == selected[None, :], axis=1)
        rel[same] = np.inf
        chosen = np.argsort(rel, kind="stable")[:N_CONTROLS]

        for j, idx in enumerate(chosen):
            combo = combos[idx]
            remaining = np.array([ch for ch in range(16) if ch not in set(combo.tolist())], dtype=np.int64)
            controls[j][concept] = np.r_[combo, remaining]
            diagnostics.append(
                {
                    "concept": concept,
                    "control": j,
                    "k": k,
                    "selected_channels": selected.tolist(),
                    "control_channels": combo.tolist(),
                    "selected_energy": target,
                    "control_energy": float(values[idx]),
                    "relative_energy_error": float(rel[idx]),
                    "energy_ratio": float(values[idx] / (target + EPS)),
                }
            )
    return controls, diagnostics


@torch.inference_mode()
def evaluate_model(model, val, test, task, block, batch, device):
    hv, lv = collect(model, val["x"], batch, device)
    ht, lt = collect(model, test["x"], batch, device)
    rankings, validation_patch_scores, validation_patch_defined = rankings_from_validation(
        model, hv, lv, task, val["concepts"], batch, device
    )
    nc = next(iter(rankings.values())).shape[0]
    vpairs = pair_tensors(task, len(hv), device)
    energy = replacement_energy(hv, vpairs, nc)

    pairs = pair_tensors(task, len(ht), device)
    b, c, g = pairs
    p0, p1 = lt[b].softmax(1), lt[c].softmax(1)
    y = c % 4
    good = (lt[b].argmax(1) == b % 4) & (lt[c].argmax(1) == y)

    contrast = rankings["contrast"]
    full = patch(model, ht, contrast, 16, pairs, batch)
    noop = patch(model, ht, contrast, 0, pairs, batch)
    assert torch.allclose(full, p1, atol=2e-5, rtol=1e-4), "Full patch identity failed"
    assert torch.allclose(noop, p0, atol=2e-5, rtol=1e-4), "No-op identity failed"

    rr = core.rng(40, block, core.TASKS.index(task))
    random_orders = [np.stack([rr.permutation(16) for _ in range(nc)]) for _ in range(N_CONTROLS)]

    random_by_k = {}
    for k in KS:
        vals = [measure(patch(model, ht, o, k, pairs, batch), p0, p1, y, good) for o in random_orders]
        random_by_k[k] = vals

    rows = []
    matching = []
    for method, order in rankings.items():
        available = method != "validation_patch" or all(validation_patch_defined)
        for k in KS:
            selected = measure(patch(model, ht, order, k, pairs, batch), p0, p1, y, good)
            rv = random_by_k[k]
            random_fidelity = float(np.mean([x["fidelity"] for x in rv])) if rv[0]["fidelity"] is not None else None
            row = {
                "method": method,
                "k": k,
                "selection_valid": bool(available),
                **selected,
                "random_fidelity": random_fidelity,
                "random_cf_accuracy": float(np.mean([x["cf_accuracy"] for x in rv])),
                "causal_usefulness_random": (
                    selected["fidelity"] - random_fidelity
                    if available and selected["fidelity"] is not None and random_fidelity is not None
                    else None
                ),
                "energy_matched_fidelity": None,
                "energy_matched_cf_accuracy": None,
                "causal_usefulness_energy": None,
            }

            if k in MATCH_KS:
                controls, diag = energy_matched_orders(order, energy, k)
                for d in diag:
                    d["method"] = method
                matching.extend(diag)
                ev = [measure(patch(model, ht, o, k, pairs, batch), p0, p1, y, good) for o in controls]
                ef = float(np.mean([x["fidelity"] for x in ev])) if ev[0]["fidelity"] is not None else None
                row["energy_matched_fidelity"] = ef
                row["energy_matched_cf_accuracy"] = float(np.mean([x["cf_accuracy"] for x in ev]))
                row["causal_usefulness_energy"] = (
                    selected["fidelity"] - ef
                    if available and selected["fidelity"] is not None and ef is not None
                    else None
                )
            rows.append(row)

    return {
        "test_acc": float((lt.argmax(1).cpu().numpy() == test["y"]).mean()),
        "alignment": core.alignment(model.conv.weight.detach().cpu().numpy(), core.bank()),
        "rows": rows,
        "rankings": {k: v.tolist() for k, v in rankings.items()},
        "validation_patch_scores": validation_patch_scores.tolist(),
        "validation_patch_defined": validation_patch_defined,
        "validation_channel_replacement_energy": energy.tolist(),
        "matching": matching,
        "denominator": float((p1 - p0).square().sum()),
        "pair_count": len(b),
        "full_patch_max_error": float((full - p1).abs().max()),
        "noop_max_error": float((noop - p0).abs().max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/exhaustive_robustness_001/training")
    p.add_argument("--output", default="results/exhaustive_robustness_001/evaluation")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent CPU fallback is used.")

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    inp, out = Path(args.input).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads((inp / "design.json").read_text())
    expected = design["blocks"] * len(design["tasks"]) * len(design["architectures"]) * len(design["profiles"])
    if not (inp / "GRID_COMPLETE.json").exists():
        raise SystemExit("Training grid is not marked complete.")

    device = torch.device(args.device)
    core.ROOT = inp
    eval_design = {
        "training_design_sha256": sha256(inp / "design.json"),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "evaluator_sha256": sha256(Path(__file__)),
        "core_sha256": sha256(Path(core.__file__)),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "batch_size": args.batch_size,
        "budgets": list(KS),
        "energy_matched_budgets": list(MATCH_KS),
        "methods": list(METHODS),
        "controls": N_CONTROLS,
        "expected_models": expected,
    }
    design_path = out / "design.json"
    if design_path.exists():
        if json.loads(design_path.read_text()) != eval_design:
            raise SystemExit("Existing evaluation design differs. Use a new --output root.")
    else:
        atomic_json(design_path, eval_design)

    jobs = []
    for block in range(design["start_block"], design["start_block"] + design["blocks"]):
        for task in design["tasks"]:
            for arch in design["architectures"]:
                for profile in design["profiles"]:
                    checkpoint = inp / "runs" / task / arch / f"block{block:04d}" / profile / f"epoch_{design['epochs']:04d}.pt"
                    jobs.append((task, arch, block, profile, checkpoint))

    for i, (task, arch, block, profile, checkpoint) in enumerate(jobs, 1):
        target = out / "runs" / task / arch / f"block{block:04d}" / profile / f"epoch_{design['epochs']:04d}.json"
        if target.exists():
            print(f"SKIP [{i}/{len(jobs)}] {task} {arch} block={block} {profile}", flush=True)
            continue
        if not checkpoint.exists():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")

        model = core.Network(arch)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.to(device).eval()

        data = {}
        for split in ("val", "test"):
            pth = inp / "data" / task / f"block{block:02d}" / f"{split}.npz"
            with np.load(pth) as z:
                data[split] = {k: z[k] for k in ("x", "y", "concepts")}

        result = evaluate_model(model, data["val"], data["test"], task, block, args.batch_size, device)
        result.update(
            {
                "task": task,
                "architecture": arch,
                "block": block,
                "profile": profile,
                "epoch": design["epochs"],
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        atomic_json(target, result)
        print(f"DONE [{i}/{len(jobs)}] {task} {arch} block={block} {profile}", flush=True)
        del model, saved
        if args.device == "cuda":
            torch.cuda.empty_cache()

    found = list((out / "runs").glob("*/*/block*/*/epoch_*.json"))
    if len(found) != expected:
        raise SystemExit(f"Incomplete evaluation grid: {len(found)} / {expected} models.")
    atomic_json(out / "GRID_COMPLETE.json", {"planned": expected, "complete": len(found)})
    print("Evaluation grid complete:", out, flush=True)


if __name__ == "__main__":
    main()
