"""Total-variation denoising (Person B). LIBRARY implementation, labelled.

Library: scikit-image `denoise_tv_chambolle(image, weight, eps, max_num_iter, channel_axis=None)`
(Chambolle's projection algorithm, isotropic TV). Record the exact objective and `weight`
convention from the INSTALLED version's docstring in the report; do not mix with
`denoise_tv_bregman`, whose weight convention differs.
Config: weight, eps (stopping tolerance), max_num_iter. skimage emits no convergence warning;
the iteration cap is the limit, so keep max_num_iter large enough for eps (default 200).
"""
from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from common.io import check_image
from common.registry import config_id as _cid

METHOD = "tv"


def denoise(image: np.ndarray, config: dict) -> np.ndarray:
    check_image(image)
    out = denoise_tv_chambolle(image.astype(np.float64), weight=float(config["weight"]),
                               eps=float(config.get("eps", 2e-4)),
                               max_num_iter=int(config.get("max_num_iter", 200)), channel_axis=None)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def config_id(config: dict) -> str:
    return _cid(config)
