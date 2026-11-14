"""Shared inference helper: reflect-pad to a multiple, run, subtract the residual, crop back."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def restore(model: torch.nn.Module, noisy: torch.Tensor, pad_multiple: int = 1,
            clip: bool = True) -> torch.Tensor:
    """noisy: (B,1,H,W) in [0,1]. Returns restored (B,1,H,W). The model is blind (no sigma)."""
    h, w = noisy.shape[-2:]
    ph, pw = (-h) % pad_multiple, (-w) % pad_multiple
    x = F.pad(noisy, (0, pw, 0, ph), mode="reflect") if (ph or pw) else noisy
    residual = model(x)
    out = (x - residual)[..., :h, :w]
    return out.clamp(0.0, 1.0) if clip else out
