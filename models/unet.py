"""Compact residual U-Net (Person B). Architectural ADAPTATION of Ronneberger et al. (2015), which
addressed biomedical segmentation, not denoising.

3 downsamplings (4 resolution levels), channel widths (16, 32, 64, 128), double 3x3 conv blocks with
GroupNorm + ReLU, ConvTranspose upsampling, skip connections by concatenation, one-channel noise
residual head (restored = noisy - prediction). Input H and W must be multiples of 8: the wrapper
pads (reflect) and crops back (see common/dl_infer.py).
Differs from DnCNN in depth, skip connections and normalisation: do not attribute result
differences to "architecture type" alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LEVELS = 3                       # number of 2x downsamplings
PAD_MULTIPLE = 2 ** LEVELS


def _block(cin: int, cout: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True))


class ResUNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), groups: int = 8):
        super().__init__()
        assert len(widths) == LEVELS + 1
        groups = min(groups, widths[0])
        self.enc = nn.ModuleList()
        cin = 1
        for w in widths[:-1]:
            self.enc.append(_block(cin, w, groups))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.mid = _block(widths[-2], widths[-1], groups)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w_hi, w_lo in zip(widths[:0:-1], widths[-2::-1]):
            self.up.append(nn.ConvTranspose2d(w_hi, w_lo, 2, stride=2))
            self.dec.append(_block(2 * w_lo, w_lo, groups))
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.widths = tuple(widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # returns the predicted NOISE residual
        assert x.shape[-1] % PAD_MULTIPLE == 0 and x.shape[-2] % PAD_MULTIPLE == 0, \
            f"input size {tuple(x.shape[-2:])} must be a multiple of {PAD_MULTIPLE}"
        skips = []
        h = x
        for blk in self.enc:
            h = blk(h)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h)
        for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), sk], dim=1))
        return self.head(h)


def build(cfg: dict) -> ResUNet:
    return ResUNet(widths=tuple(cfg.get("widths", (16, 32, 64, 128))), groups=int(cfg.get("groups", 8)))
