from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.templates.primitives import TemplateSpec, make_template_bank, template_names

@dataclass
class KernelSummary:
    # kernels are expected shape (C, 1, k, k) or (C, k, k)
    mean: np.ndarray
    std: np.ndarray
    per_run: np.ndarray  # (R, C, k, k)
    order_note: str

def _squeeze(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 4 and W.shape[1] == 1:
        return W[:, 0]
    if W.ndim == 3:
        return W
    raise ValueError(f"Unexpected kernel array shape: {W.shape}")

def canonical_order_by_templates(
    W: np.ndarray,
    best_template: np.ndarray,
    best_score: np.ndarray,
    templates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder + sign-align kernels using (best_template, best_score).

    Returns:
      W_can: (C,k,k) kernels in deterministic order
      perm:  permutation indices (C,)
    """
    W2 = _squeeze(W)
    C = W2.shape[0]
    bt = np.asarray(best_template).astype(int).reshape(-1)[:C]
    bs = np.asarray(best_score).astype(np.float32).reshape(-1)[:C]

    # sign alignment: flip kernel to have positive dot with its best template
    T = _squeeze(templates)
    for i in range(C):
        t = int(bt[i]) if bt.size > i else 0
        if 0 <= t < T.shape[0]:
            dot = float((W2[i] * T[t]).sum())
            if dot < 0:
                W2[i] = -W2[i]

    # deterministic ordering: primary by template id, secondary by score descending
    keys = [(int(bt[i]) if bt.size > i else 10**9, -float(bs[i]) if bs.size > i else 0.0, i) for i in range(C)]
    perm = np.array([k[2] for k in sorted(keys)], dtype=int)
    return W2[perm], perm

def mean_kernels_across_runs(
    Ws: Sequence[np.ndarray],
    best_templates: Optional[Sequence[np.ndarray]] = None,
    best_scores: Optional[Sequence[np.ndarray]] = None,
    template_spec: Optional[TemplateSpec] = None,
    do_canonicalize: bool = True,
) -> KernelSummary:
    """Compute mean/std kernels across runs with optional canonicalization.

    Note: averaging kernels is only meaningful if channel ordering is aligned.
    We provide a *template-based canonicalization* that is well-motivated for
    template_init, and at least deterministic for random.
    """
    if template_spec is None:
        template_spec = TemplateSpec()
    templates = make_template_bank(template_spec)

    per = []
    order_note = "raw channel index ordering"
    for r, W in enumerate(Ws):
        W2 = _squeeze(W)
        if do_canonicalize and best_templates is not None and best_scores is not None:
            W2, _ = canonical_order_by_templates(W2, best_templates[r], best_scores[r], templates)
            order_note = "ordered by (best_template, -best_score) with sign alignment"
        per.append(W2)

    per_run = np.stack(per, axis=0).astype(np.float32)  # (R,C,k,k)
    mean = per_run.mean(axis=0)
    std = per_run.std(axis=0, ddof=1) if per_run.shape[0] > 1 else np.zeros_like(mean)
    return KernelSummary(mean=mean, std=std, per_run=per_run, order_note=order_note)

def get_template_bank(template_spec: Optional[TemplateSpec] = None) -> Tuple[np.ndarray, List[str]]:
    if template_spec is None:
        template_spec = TemplateSpec()
    T = make_template_bank(template_spec)
    names = template_names(template_spec)
    return T, names
