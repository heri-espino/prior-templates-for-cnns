from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset

SHAPES = ["line", "circle", "triangle", "square"]
LABEL2NAME = {i: n for i, n in enumerate(SHAPES)}
NAME2LABEL = {n: i for i, n in enumerate(SHAPES)}

@dataclass(frozen=True)
class FactorRanges:
    theta_min: float
    theta_max: float
    tx_min: float
    tx_max: float
    ty_min: float
    ty_max: float
    scale_min: float
    scale_max: float
    thick_min: float
    thick_max: float
    noise_std_min: float
    noise_std_max: float
    occ_prob: float
    occ_size_min: float
    occ_size_max: float
    aa_min: int
    aa_max: int

@dataclass(frozen=True)
class SplitSpec:
    name: str
    n: int
    ranges: FactorRanges
    seed: int

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _draw_regular_polygon(draw: ImageDraw.ImageDraw, center: Tuple[float, float], radius: float, sides: int, angle: float, width: int) -> None:
    cx, cy = center
    pts: List[Tuple[float, float]] = []
    for i in range(sides):
        a = angle + i * 2 * math.pi / sides
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    draw.line(pts + [pts[0]], fill=0, width=width, joint="curve")

def render_shape(
    shape: str,
    size: int,
    theta: float,
    tx: float,
    ty: float,
    scale: float,
    thickness: float,
    noise_std: float,
    occ_prob: float,
    occ_size: Tuple[float, float],
    aa: int,
    blur_sigma: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    assert shape in SHAPES
    rng = _rng(seed if seed is not None else 0)

    S_hi = int(size * aa)
    img = Image.new("L", (S_hi, S_hi), color=255)
    draw = ImageDraw.Draw(img)

    cx, cy = S_hi / 2.0 + tx * aa, S_hi / 2.0 + ty * aa
    base = min(S_hi, S_hi) * 0.35 * scale
    w = max(1, int(round(thickness * aa)))

    if shape == "line":
        length = base * 2.0
        x0 = cx - (length/2)*math.cos(theta)
        y0 = cy - (length/2)*math.sin(theta)
        x1 = cx + (length/2)*math.cos(theta)
        y1 = cy + (length/2)*math.sin(theta)
        draw.line([(x0,y0),(x1,y1)], fill=0, width=w)
    elif shape == "circle":
        r = base
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=0, width=w)
    elif shape == "triangle":
        _draw_regular_polygon(draw, (cx,cy), base, sides=3, angle=theta, width=w)
    elif shape == "square":
        _draw_regular_polygon(draw, (cx,cy), base, sides=4, angle=theta, width=w)

    if rng.random() < occ_prob:
        occ_w, occ_h = occ_size
        ow = int(round(occ_w * aa))
        oh = int(round(occ_h * aa))
        ox = int(rng.uniform(0, S_hi - max(1, ow)))
        oy = int(rng.uniform(0, S_hi - max(1, oh)))
        ImageDraw.Draw(img).rectangle([ox, oy, ox+ow, oy+oh], fill=255)

    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma * aa))

    img = img.resize((size, size), resample=Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = 1.0 - arr
    if noise_std > 0:
        arr = arr + rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)

class ShapesDataset(Dataset):
    def __init__(self, split: SplitSpec, image_size: int = 64, return_meta: bool = False):
        self.split = split
        self.image_size = image_size
        self.return_meta = return_meta

        rng = _rng(split.seed)
        n = split.n
        self.x = np.zeros((n, image_size, image_size), dtype=np.float32)
        self.y = np.zeros((n,), dtype=np.int64)
        self.meta: List[Dict[str, float]] = []

        R = split.ranges
        for i in range(n):
            yi = int(rng.integers(0, len(SHAPES)))
            shape = SHAPES[yi]
            theta = float(rng.uniform(R.theta_min, R.theta_max))
            tx = float(rng.uniform(R.tx_min, R.tx_max))
            ty = float(rng.uniform(R.ty_min, R.ty_max))
            scale = float(rng.uniform(R.scale_min, R.scale_max))
            thick = float(rng.uniform(R.thick_min, R.thick_max))
            noise = float(rng.uniform(R.noise_std_min, R.noise_std_max))
            aa = int(rng.integers(R.aa_min, R.aa_max + 1))
            occ_w = float(rng.uniform(R.occ_size_min, R.occ_size_max))
            occ_h = float(rng.uniform(R.occ_size_min, R.occ_size_max))

            img = render_shape(
                shape=shape,
                size=image_size,
                theta=theta,
                tx=tx,
                ty=ty,
                scale=scale,
                thickness=thick,
                noise_std=noise,
                occ_prob=float(R.occ_prob),
                occ_size=(occ_w, occ_h),
                aa=aa,
                blur_sigma=0.0,
                seed=int(rng.integers(0, 1_000_000_000)),
            )
            self.x[i] = img
            self.y[i] = yi
            if self.return_meta:
                self.meta.append(dict(theta=theta, tx=tx, ty=ty, scale=scale, thickness=thick, noise_std=noise, aa=float(aa)))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).unsqueeze(0)
        y = int(self.y[idx])
        if self.return_meta:
            return x, y, self.meta[idx]
        return x, y

def default_splits(seed: int, train_size: int, val_size: int, test_size: int) -> Dict[str, SplitSpec]:
    train_ranges = FactorRanges(
        theta_min=0.0, theta_max=math.pi/4,
        tx_min=-6, tx_max=6, ty_min=-6, ty_max=6,
        scale_min=0.8, scale_max=1.2,
        thick_min=1.0, thick_max=3.0,
        noise_std_min=0.0, noise_std_max=0.10,
        occ_prob=0.10, occ_size_min=6.0, occ_size_max=18.0,
        aa_min=4, aa_max=8,
    )
    ood_rot = FactorRanges(**{**train_ranges.__dict__, "theta_min": math.pi/4, "theta_max": 2*math.pi})
    ood_thick = FactorRanges(**{**train_ranges.__dict__, "thick_min": 4.0, "thick_max": 6.0})
    ood_occ = FactorRanges(**{**train_ranges.__dict__, "occ_prob": 0.40})

    return {
        "train": SplitSpec("train", train_size, train_ranges, seed=seed + 1),
        "val": SplitSpec("val", val_size, train_ranges, seed=seed + 2),
        "test": SplitSpec("test", test_size, train_ranges, seed=seed + 3),
        "ood_rot": SplitSpec("ood_rot", test_size, ood_rot, seed=seed + 4),
        "ood_thick": SplitSpec("ood_thick", test_size, ood_thick, seed=seed + 5),
        "ood_occ": SplitSpec("ood_occ", test_size, ood_occ, seed=seed + 6),
    }
