from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
import pandas as pd

@dataclass
class RunRecord:
    run_dir: Path
    regime: str
    seed: int
    train_log: pd.DataFrame
    metrics: Dict[str, Any]
    align_best: np.ndarray
    delta_cond: np.ndarray
    patch1_success: np.ndarray
    patch1_shift: np.ndarray
    patchk_success: np.ndarray
    patchk_shift: np.ndarray
    drift_cos: np.ndarray
    drift_l2: np.ndarray
    conv_W_init: np.ndarray
    conv_W_final: np.ndarray

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def load_run(run_dir: Path) -> RunRecord:
    cfg = _read_json(run_dir / "config.json")
    regime = cfg["args"]["regime"]
    seed = int(cfg["args"]["seed"])

    train_rows = _read_json(run_dir / "train_log.json")["rows"]
    train_log = pd.DataFrame(train_rows)

    metrics = _read_json(run_dir / "metrics.json")

    align_best = np.load(run_dir / "alignment_best_score.npy")
    delta_cond = np.load(run_dir / "delta_cond.npy")
    patch1_success = np.load(run_dir / "patch_single_success.npy")
    patch1_shift = np.load(run_dir / "patch_single_shift.npy")
    patchk_success = np.load(run_dir / "patch_topk_success.npy")
    patchk_shift = np.load(run_dir / "patch_topk_shift.npy")
    drift_cos = np.load(run_dir / "drift_cosine.npy")
    drift_l2 = np.load(run_dir / "drift_l2.npy")

    conv_W_init = np.load(run_dir / "conv_W_init.npy") if (run_dir / "conv_W_init.npy").exists() else np.zeros((0,), dtype=np.float32)
    conv_W_final = np.load(run_dir / "conv_W_final.npy") if (run_dir / "conv_W_final.npy").exists() else np.zeros((0,), dtype=np.float32)

    return RunRecord(
        run_dir=run_dir,
        regime=regime,
        seed=seed,
        train_log=train_log,
        metrics=metrics,
        align_best=align_best,
        delta_cond=delta_cond,
        patch1_success=patch1_success,
        patch1_shift=patch1_shift,
        patchk_success=patchk_success,
        patchk_shift=patchk_shift,
        drift_cos=drift_cos,
        drift_l2=drift_l2,
        conv_W_init=conv_W_init,
        conv_W_final=conv_W_final,
    )

def discover_runs(runs_root: Path) -> List[Path]:
    # expects leaf dirs containing metrics.json
    return sorted([p.parent for p in runs_root.rglob("metrics.json")])

def group_by_regime(records: Sequence[RunRecord]) -> Dict[str, List[RunRecord]]:
    out: Dict[str, List[RunRecord]] = {}
    for r in records:
        out.setdefault(r.regime, []).append(r)
    for k in out:
        out[k] = sorted(out[k], key=lambda rr: rr.seed)
    return out

def _stack_learning_curves(records: Sequence[RunRecord], key: str = "val_acc") -> Tuple[np.ndarray, np.ndarray]:
    # returns epochs, values [n_runs, n_epochs]
    curves = []
    for r in records:
        df = r.train_log.sort_values("epoch")
        curves.append(df[key].to_numpy(dtype=np.float32))
    # pad to same length
    maxlen = max(len(c) for c in curves)
    X = np.full((len(curves), maxlen), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        X[i, :len(c)] = c
    epochs = np.arange(1, maxlen + 1, dtype=np.int32)
    return epochs, X

def aggregate_curve(records: Sequence[RunRecord], key: str = "val_acc") -> Dict[str, np.ndarray]:
    epochs, X = _stack_learning_curves(records, key=key)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mean)
    return {"epochs": epochs, "mean": mean, "std": std, "all": X}

def aggregate_vector(records: Sequence[RunRecord], attr: str) -> Dict[str, np.ndarray]:
    # attr is a 1D vector per run (e.g. patch1_success)
    X = np.stack([getattr(r, attr) for r in records], axis=0).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mean)
    return {"mean": mean, "std": std, "all": X}

def aggregate_scalar_metric(records: Sequence[RunRecord], metric_key: str) -> Dict[str, float]:
    vals = []
    for r in records:
        v = r.metrics.get(metric_key, None)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    arr = np.array(vals, dtype=np.float32) if vals else np.array([], dtype=np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}

def time_to_threshold(train_log: "pd.DataFrame", key: str, thr: float) -> float:
    df = train_log.sort_values("epoch")
    hit = df[df[key] >= thr]
    if len(hit) == 0:
        return float("inf")
    return float(hit.iloc[0]["epoch"])

def aggregate_time_to_threshold(records: Sequence[RunRecord], key: str = "val_acc", thr: float = 0.98) -> Dict[str, float]:
    ts = np.array([time_to_threshold(r.train_log, key=key, thr=thr) for r in records], dtype=np.float32)
    finite = ts[np.isfinite(ts)]
    if finite.size == 0:
        return {"mean": float("inf"), "std": float("nan"), "n": 0}
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return {"mean": float(finite.mean()), "std": std, "n": int(finite.size)}
