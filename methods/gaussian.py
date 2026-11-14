"""Gaussian smoothing (Person A). Linear baseline.

Library: scipy.ndimage.gaussian_filter (truncate=4.0, border mode 'reflect'). The existing
from-scratch implementation may replace the body of `denoise` unchanged behind this interface.
Config: sigma_px (spatial scale in pixels).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from common.io import check_image
from common.registry import config_id as _cid

METHOD = "gaussian"


def denoise(image: np.ndarray, config: dict) -> np.ndarray:
    check_image(image)
    out = ndi.gaussian_filter(image.astype(np.float64), sigma=float(config["sigma_px"]),
                              mode="reflect", truncate=4.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def config_id(config: dict) -> str:
    return _cid(config)
