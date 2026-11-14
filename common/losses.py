"""Pixel (P) and pixel + Sobel gradient (G) losses. Operates on the RESTORED image (unclipped)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradLoss(nn.Module):
    """L1(Sobel_x) + L1(Sobel_y) between pred and target. Kernels divided by 8.
    Border: replicate padding then 'valid' conv (matches scipy mode='nearest')."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def grads(self, x: torch.Tensor):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.kx), F.conv2d(x, self.ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gxp, gyp = self.grads(pred)
        gxt, gyt = self.grads(target)
        return F.l1_loss(gxp, gxt) + F.l1_loss(gyp, gyt)


def total_loss(restored, clean, variant: str, lam: float, sobel: SobelGradLoss) -> torch.Tensor:
    """P: L1 only.  G: L1 + lam * (Sobel-x L1 + Sobel-y L1)."""
    pix = F.l1_loss(restored, clean)
    if variant == "P":
        return pix
    if variant == "G":
        return pix + lam * sobel(restored, clean)
    raise ValueError(f"unknown loss variant {variant!r}")
