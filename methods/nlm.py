"""Non-Local Means (Person A). LIBRARY implementation, labelled.

Library: scikit-image `denoise_nl_means(image, patch_size, patch_distance, h, sigma, fast_mode)`.
Config: patch_size, patch_distance, h (filtering strength on the [0,1] scale), sigma_255 (told
sigma, converted to /255 inside; allowed because classical methods are told sigma), fast_mode.
"""
from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_nl_means

from common.io import check_image
from common.registry import config_id as _cid

METHOD = "nlm"


def denoise(image: np.ndarray, config: dict) -> np.ndarray:
    check_image(image)
    out = denoise_nl_means(
        image.astype(np.float64), patch_size=int(config.get("patch_size", 5)),
        patch_distance=int(config.get("patch_distance", 6)), h=float(config["h"]),
        sigma=float(config.get("sigma_255", 0.0)) / 255.0, fast_mode=bool(config.get("fast_mode", True)),
        channel_axis=None)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def config_id(config: dict) -> str:
    return _cid(config)
