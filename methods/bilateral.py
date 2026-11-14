"""Bilateral filter (Person B), from scratch, vectorised over window offsets.

weight(p, q) = exp(-|p-q|^2 / (2 sigma_s^2)) * exp(-(I(p)-I(q))^2 / (2 sigma_r^2)), normalised per pixel.
Config (stored units):
  sigma_s_px   spatial scale in pixels
  sigma_r_255  range scale on the 0..255 intensity scale; converted to /255 inside this wrapper
  radius_px    neighbourhood support (window is (2r+1)^2); default ceil(3 * sigma_s_px)
Border handling: reflect padding. Runtime grows with radius^2.
"""
from __future__ import annotations

import math

import numpy as np

from common.io import check_image
from common.registry import config_id as _cid

METHOD = "bilateral"


def denoise(image: np.ndarray, config: dict) -> np.ndarray:
    check_image(image)
    ss = float(config["sigma_s_px"])
    sr = float(config["sigma_r_255"]) / 255.0          # 0..255 units -> [0,1] interface scale
    r = int(config.get("radius_px", math.ceil(3 * ss)))
    x = image.astype(np.float64)
    h, w = x.shape
    pad = np.pad(x, r, mode="reflect")
    num = np.zeros_like(x)
    den = np.zeros_like(x)
    inv2sr = 1.0 / (2.0 * sr * sr)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = math.exp(-(dy * dy + dx * dx) / (2.0 * ss * ss))
            shifted = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            wgt = ws * np.exp(-((shifted - x) ** 2) * inv2sr)
            num += wgt * shifted
            den += wgt
    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def config_id(config: dict) -> str:
    return _cid(config)
