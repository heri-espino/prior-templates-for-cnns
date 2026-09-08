from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

@dataclass(frozen=True)
class TemplateSpec:
    k: int = 9
    sigma: float = 2.0
    n_edges: int = 8
    n_corners: int = 4
    n_rings: int = 4
    ring_radii: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)

def _grid(k: int) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.arange(-(k//2), k//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    return xx, yy

def _normalize_zero_mean(unit: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    unit = unit.astype(np.float32)
    unit = unit - unit.mean()
    denom = float(np.sqrt((unit**2).sum()) + eps)
    return unit / denom

def oriented_edge(k: int, theta: float, sigma: float) -> np.ndarray:
    x, y = _grid(k)
    ct, st = math.cos(theta), math.sin(theta)
    x_p =  ct * x + st * y
    y_p = -st * x + ct * y
    g  = np.exp(-(x_p**2 + y_p**2) / (2 * sigma**2)).astype(np.float32)
    f  = x_p * g
    return _normalize_zero_mean(f)

def corner(k: int, theta: float, sigma: float) -> np.ndarray:
    e1 = oriented_edge(k, theta, sigma)
    e2 = oriented_edge(k, theta + math.pi/2, sigma)
    return _normalize_zero_mean(e1 + e2)

def ring(k: int, radius: float, sigma: float) -> np.ndarray:
    x, y = _grid(k)
    r = np.sqrt(x**2 + y**2).astype(np.float32)
    g1 = np.exp(-((r - radius)**2) / (2 * sigma**2)).astype(np.float32)
    g2 = np.exp(-((r - (radius + 1.0))**2) / (2 * sigma**2)).astype(np.float32)
    return _normalize_zero_mean(g1 - g2)

def make_template_bank(spec: TemplateSpec) -> np.ndarray:
    templates: List[np.ndarray] = []
    for i in range(spec.n_edges):
        templates.append(oriented_edge(spec.k, i * (math.pi/spec.n_edges), spec.sigma))
    for i in range(spec.n_corners):
        templates.append(corner(spec.k, i * (math.pi/spec.n_corners), spec.sigma))
    for rad in spec.ring_radii[:spec.n_rings]:
        templates.append(ring(spec.k, rad, sigma=max(0.8, spec.sigma/2)))
    return np.stack(templates, axis=0).astype(np.float32)

def template_names(spec: TemplateSpec) -> List[str]:
    names: List[str] = []
    for i in range(spec.n_edges):
        names.append(f"edge_{i:02d}")
    for i in range(spec.n_corners):
        names.append(f"corner_{i:02d}")
    for i in range(spec.n_rings):
        names.append(f"ring_{i:02d}")
    return names
