"""DnCNN (Person A): residual CNN. Conv-ReLU, 15x (Conv-BN-ReLU), Conv; predicts the noise.

Default: depth 17, width 64 (~0.56M params), BatchNorm, receptive field 35x35. Fully convolutional,
so no padding is needed at inference (pad_multiple = 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PAD_MULTIPLE = 1


class DnCNN(nn.Module):
    def __init__(self, depth: int = 17, width: int = 64):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, 1, 3, padding=1, bias=False))
        self.body = nn.Sequential(*layers)
        self.depth, self.width = depth, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # returns the predicted NOISE residual
        return self.body(x)


def build(cfg: dict) -> DnCNN:
    return DnCNN(depth=int(cfg.get("depth", 17)), width=int(cfg.get("width", 64)))
