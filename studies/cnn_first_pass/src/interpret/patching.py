from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

@dataclass
class SingleChannelPatchingResult:
    success_rate: np.ndarray     # (C,)
    mean_margin_shift: np.ndarray  # (C,)
    summary: Dict[str, float]

@dataclass
class TopKPatchingResult:
    ks: List[int]
    success_rate: np.ndarray       # (len(ks),)
    mean_margin_shift: np.ndarray  # (len(ks),)
    summary: Dict[str, float]

@torch.no_grad()
def _collect_logits_z_y(model, loader, device: torch.device, max_items: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all, z_all, y_all = [], [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits, z = model(x)
        logits_all.append(logits.detach().cpu())
        z_all.append(z.detach().cpu())
        y_all.append(y.detach().cpu())
        n += x.shape[0]
        if n >= max_items:
            break
    L = torch.cat(logits_all, dim=0)[:max_items]
    Z = torch.cat(z_all, dim=0)[:max_items]
    Y = torch.cat(y_all, dim=0)[:max_items]
    return L, Z, Y

def _make_mismatched_pairs(Y: torch.Tensor, num_pairs: int, num_classes: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (idx_s, idx_t) with y_s != y_t."""
    g = torch.Generator().manual_seed(seed)
    idx_by_class = [torch.where(Y == c)[0] for c in range(num_classes)]
    # ensure non-empty classes
    available = [c for c in range(num_classes) if len(idx_by_class[c]) > 0]
    if len(available) < 2:
        raise ValueError("Need at least 2 non-empty classes to form mismatched pairs.")

    idx_s = torch.empty((num_pairs,), dtype=torch.long)
    idx_t = torch.empty((num_pairs,), dtype=torch.long)
    for p in range(num_pairs):
        c_s = int(available[torch.randint(0, len(available), (1,), generator=g)])
        # choose a different class
        c_t = c_s
        while c_t == c_s:
            c_t = int(available[torch.randint(0, len(available), (1,), generator=g)])
        s_pool = idx_by_class[c_s]
        t_pool = idx_by_class[c_t]
        idx_s[p] = s_pool[torch.randint(0, len(s_pool), (1,), generator=g)]
        idx_t[p] = t_pool[torch.randint(0, len(t_pool), (1,), generator=g)]
    return idx_s, idx_t

@torch.no_grad()
def single_channel_patching(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    max_items: int = 2000,
    num_pairs: int = 1000,
    seed: int = 0,
) -> SingleChannelPatchingResult:
    """Controlled single-channel patching at pooled vector z.

    For mismatched pairs (s,t) with y_s != y_t:
      base_margin = logits_t[y_s] - logits_t[y_t]
      patch z_t[i] <- z_s[i] => logits_patch = logits_t + V[:,i]*(z_s[i]-z_t[i])
      margin_shift = (V[y_s,i]-V[y_t,i])*(z_s[i]-z_t[i])

    Returns per-channel:
      success_rate[i] = P(margin_shift > 0)
      mean_margin_shift[i] = E[margin_shift]
    """
    L, Z, Y = _collect_logits_z_y(model, loader, device=device, max_items=max_items)
    K = int(num_classes)
    V = model.classifier.weight.detach().cpu()  # (K,C)
    C = Z.shape[1]

    idx_s, idx_t = _make_mismatched_pairs(Y, num_pairs=num_pairs, num_classes=K, seed=seed)
    y_s = Y[idx_s]
    y_t = Y[idx_t]
    z_s = Z[idx_s]  # (P,C)
    z_t = Z[idx_t]  # (P,C)

    # compute per-pair weight differences per channel: wdiff[p,i] = V[y_s,i] - V[y_t,i]
    w_s = V[y_s]  # (P,C)
    w_t = V[y_t]  # (P,C)
    wdiff = w_s - w_t
    dz = z_s - z_t
    shift = wdiff * dz  # (P,C)

    success = (shift > 0).float().mean(dim=0).numpy()
    mean_shift = shift.mean(dim=0).numpy()

    summary = {
        "mean_success": float(success.mean()),
        "max_success": float(success.max()),
        "mean_shift": float(mean_shift.mean()),
        "max_shift": float(mean_shift.max()),
        "num_pairs": int(num_pairs),
    }
    return SingleChannelPatchingResult(success_rate=success, mean_margin_shift=mean_shift, summary=summary)

def _topk_channels_per_class(delta_cond: np.ndarray, ks: Sequence[int]) -> List[List[np.ndarray]]:
    """delta_cond: (C,K). For each class c, sort channels descending by delta_cond[:,c]."""
    C, K = delta_cond.shape
    order = [np.argsort(-delta_cond[:, c]) for c in range(K)]
    topks: List[List[np.ndarray]] = []
    for k in ks:
        topks.append([order[c][:k] for c in range(K)])
    return topks

@torch.no_grad()
def topk_patching(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    delta_cond: np.ndarray,
    ks: Sequence[int] = (1, 2, 4, 8),
    max_items: int = 2000,
    num_pairs: int = 1000,
    seed: int = 0,
) -> TopKPatchingResult:
    """Patch the top-k channels for the *source class* (ranked by delta_cond).

    For each mismatched pair (s,t):
      S = top_k_channels[source_class]
      logits_patch = logits_t + V[:,S] @ (z_s[S]-z_t[S])
      margin_shift = (logits_patch[y_s]-logits_patch[y_t]) - (logits_t[y_s]-logits_t[y_t])

    Reports success and effect size aggregated over pairs.
    """
    L, Z, Y = _collect_logits_z_y(model, loader, device=device, max_items=max_items)
    K = int(num_classes)
    V = model.classifier.weight.detach().cpu()  # (K,C)

    idx_s, idx_t = _make_mismatched_pairs(Y, num_pairs=num_pairs, num_classes=K, seed=seed)
    y_s = Y[idx_s]
    y_t = Y[idx_t]
    z_s = Z[idx_s]  # (P,C)
    z_t = Z[idx_t]  # (P,C)
    logits_t = L[idx_t]  # (P,K)

    base = logits_t.gather(1, y_s.view(-1,1)).squeeze(1) - logits_t.gather(1, y_t.view(-1,1)).squeeze(1)

    ks = list(ks)
    topks = _topk_channels_per_class(delta_cond=delta_cond, ks=ks)

    succ = []
    msh = []
    for k_idx, k in enumerate(ks):
        shifts = torch.empty((num_pairs,), dtype=torch.float32)
        for p in range(num_pairs):
            c = int(y_s[p].item())
            S = topks[k_idx][c]  # numpy array of channels
            S_t = torch.from_numpy(S).long()
            dz = (z_s[p, S_t] - z_t[p, S_t]).view(-1, 1)  # (k,1)
            # delta logits: (K,k)@(k,1) -> (K,1)
            dlogits = (V[:, S_t] @ dz).view(-1)  # (K,)
            lt = logits_t[p] + dlogits
            patched = lt[y_s[p]] - lt[y_t[p]]
            shifts[p] = patched - base[p]

        succ.append(float((shifts > 0).float().mean().item()))
        msh.append(float(shifts.mean().item()))

    succ_np = np.array(succ, dtype=np.float32)
    msh_np = np.array(msh, dtype=np.float32)

    summary = {
        "success@maxk": float(succ_np.max()),
        "mean_success": float(succ_np.mean()),
        "mean_shift": float(msh_np.mean()),
        "num_pairs": int(num_pairs),
    }
    return TopKPatchingResult(ks=ks, success_rate=succ_np, mean_margin_shift=msh_np, summary=summary)
