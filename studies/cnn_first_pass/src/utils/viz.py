from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_kernels_grid(
    kernels: np.ndarray,
    outpath: Path,
    title: str = "",
    max_kernels: int = 32,
    cols: int = 8,
) -> None:
    """Save conv kernels as an image grid.

    kernels: (C, 1, k, k) or (C, k, k)
    """
    if kernels.ndim == 4:
        kernels = kernels[:,0]
    C = kernels.shape[0]
    n = min(C, max_kernels)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        w = kernels[i]
        # normalize for visualization (not for metrics)
        w = w - w.mean()
        denom = np.max(np.abs(w)) + 1e-8
        w = w / denom
        ax.imshow(w, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"k{i:02d}", fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
