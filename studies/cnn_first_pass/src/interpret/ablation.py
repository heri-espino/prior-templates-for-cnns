from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

@dataclass
class ConditionalAblationResult:
    # delta_cond: (C, K) where entry [i,c] is effect of ablating channel i on correct logit for class c
    delta_cond: np.ndarray
    mean_z_by_class: np.ndarray
    specificity: np.ndarray
    summary: Dict[str, float]

@torch.no_grad()
def collect_z_y(model, loader, device: torch.device, max_items: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    zs = []
    ys = []
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits, z = model(x)
        zs.append(z.detach().cpu())
        ys.append(y.detach().cpu())
        n += x.shape[0]
        if max_items is not None and n >= max_items:
            break
    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    if max_items is not None:
        Z = Z[:max_items]
        Y = Y[:max_items]
    return Z, Y

@torch.no_grad()
def class_conditional_ablation(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    max_items: Optional[int] = None,
) -> ConditionalAblationResult:
    """Class-conditional 'ablation' for a linear head.

    Model: logits = V z + b. If we ablate z_i -> 0, then:
      logits_k - logits_k(ablated i) = V[k,i] * z_i.
    Conditioning on y=c and focusing on correct logit:
      Δ_cond[i,c] = E[ V[c,i] * z_i | y=c ] = V[c,i] * E[z_i | y=c].

    Returns delta_cond with shape (C,K), where column c is class c.
    """
    Z, Y = collect_z_y(model, loader, device=device, max_items=max_items)
    V = model.classifier.weight.detach().cpu()  # (K,C)
    C = Z.shape[1]
    K = int(num_classes)

    mean_z = torch.zeros((K, C), dtype=torch.float32)
    counts = torch.zeros((K,), dtype=torch.float32)
    for c in range(K):
        mask = (Y == c)
        if mask.any():
            mean_z[c] = Z[mask].mean(dim=0)
            counts[c] = float(mask.sum())

    # delta_cond[c,i] = V[c,i] * mean_z[c,i]
    delta_cond_c_i = (V * mean_z)  # broadcast (K,C)
    # return as (C,K) for plotting channel rows vs class cols
    delta_cond = delta_cond_c_i.T.numpy()
    mean_z_by_class = mean_z.numpy()

    # channel specificity: best class minus mean of other classes
    spec = np.zeros((C,), dtype=np.float32)
    for i in range(C):
        col = delta_cond[i]  # (K,)
        c_star = int(col.argmax())
        others = [c for c in range(K) if c != c_star]
        spec[i] = float(col[c_star] - (col[others].mean() if others else 0.0))

    summary = {
        "mean_specificity": float(spec.mean()),
        "median_specificity": float(np.median(spec)),
        "max_specificity": float(spec.max()),
        "min_specificity": float(spec.min()),
        "class_counts_min": float(counts.min().item()),
    }
    return ConditionalAblationResult(
        delta_cond=delta_cond,
        mean_z_by_class=mean_z_by_class,
        specificity=spec,
        summary=summary,
    )
