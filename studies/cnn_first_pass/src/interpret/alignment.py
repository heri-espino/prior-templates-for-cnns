from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch

@dataclass
class AlignmentResult:
    A: np.ndarray
    best_template: np.ndarray
    best_score: np.ndarray
    summary: Dict[str, float]

def cosine_matrix(filters: np.ndarray, templates: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if filters.ndim == 4:
        filters = filters[:,0]
    F = filters.reshape(filters.shape[0], -1).astype(np.float32)
    T = templates.reshape(templates.shape[0], -1).astype(np.float32)
    F = F - F.mean(axis=1, keepdims=True)
    T = T - T.mean(axis=1, keepdims=True)
    Fn = np.linalg.norm(F, axis=1, keepdims=True) + eps
    Tn = np.linalg.norm(T, axis=1, keepdims=True) + eps
    return (F / Fn) @ (T / Tn).T

def compute_alignment(conv_weight: torch.Tensor, templates: np.ndarray) -> AlignmentResult:
    W = conv_weight.detach().cpu().numpy()
    A = cosine_matrix(W, templates)
    best_template = A.argmax(axis=1)
    best_score = A.max(axis=1)
    summary = {
        "mean_best_score": float(best_score.mean()),
        "median_best_score": float(np.median(best_score)),
        "min_best_score": float(best_score.min()),
        "max_best_score": float(best_score.max()),
    }
    return AlignmentResult(A=A, best_template=best_template, best_score=best_score, summary=summary)
