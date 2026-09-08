from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Regime = Literal["random", "template_init", "frozen_templates"]

@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 64
    num_classes: int = 4
    kernel_size: int = 9
    channels: int = 16
    regime: Regime = "random"
    template_channels: int = 16

class TinyCNN(nn.Module):
    def __init__(self, cfg: ModelConfig, templates: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        pad = cfg.kernel_size // 2
        self.conv = nn.Conv2d(1, cfg.channels, kernel_size=cfg.kernel_size, padding=pad, bias=False)
        self.classifier = nn.Linear(cfg.channels, cfg.num_classes, bias=True)

        if cfg.regime != "random":
            if templates is None:
                raise ValueError("templates must be provided for template regimes")
            self.apply_templates(templates, freeze=(cfg.regime == "frozen_templates"))

    def apply_templates(self, templates: np.ndarray, freeze: bool) -> None:
        k = self.cfg.kernel_size
        if templates.shape[1:] != (k, k):
            raise ValueError(f"templates must have shape (p,{k},{k}), got {templates.shape}")
        m = min(self.cfg.template_channels, self.cfg.channels, templates.shape[0])
        W = torch.from_numpy(templates[:m]).unsqueeze(1)  # (m,1,k,k)
        with torch.no_grad():
            self.conv.weight[:m].copy_(W)

        if freeze:
            mask = torch.zeros_like(self.conv.weight)
            mask[:m] = 1.0
            self.register_buffer("_freeze_mask", mask)

            def _hook(grad):
                return grad * (1.0 - self._freeze_mask)

            self.conv.weight.register_hook(_hook)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.conv(x))
        z = torch.amax(h, dim=(2,3))  # pooled presence vector
        logits = self.classifier(z)
        return logits, z
