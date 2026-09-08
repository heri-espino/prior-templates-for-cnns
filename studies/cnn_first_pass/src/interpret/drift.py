from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch

@dataclass
class DriftResult:
    cosine: np.ndarray
    l2: np.ndarray
    summary: Dict[str, float]

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a.reshape(a.shape[0], -1).astype(np.float32)
    b = b.reshape(b.shape[0], -1).astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    return (a*b).sum(axis=1) / (an*bn)

def compute_drift(W_init: torch.Tensor, W_final: torch.Tensor) -> DriftResult:
    Wi = W_init.detach().cpu().numpy()
    Wf = W_final.detach().cpu().numpy()
    if Wi.ndim == 4:
        Wi = Wi[:,0]
        Wf = Wf[:,0]
    cos = _cosine(Wi, Wf)
    l2 = np.sqrt(((Wi - Wf)**2).reshape(Wi.shape[0], -1).sum(axis=1)).astype(np.float32)
    summary = {
        "mean_cosine": float(cos.mean()),
        "median_cosine": float(np.median(cos)),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "max_l2": float(l2.max()),
    }
    return DriftResult(cosine=cos.astype(np.float32), l2=l2, summary=summary)
