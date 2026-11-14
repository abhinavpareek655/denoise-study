"""PSNR / SSIM / MAE on 8-bit-quantized images. Peak is fixed at 1.0 on the [0,1] scale."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from common.io import quantize8
from common.protocol import PSNR_CAP_DB, SSIM_DATA_RANGE, SSIM_SIGMA, SSIM_WIN_SIZE


def q01(x01: np.ndarray) -> np.ndarray:
    """Quantize to 8-bit and return float64 in [0,1]."""
    return quantize8(x01).astype(np.float64) / 255.0


def mse(clean01, out01, mask=None) -> float:
    d = (q01(clean01) - q01(out01)) ** 2
    return float(d.mean() if mask is None else d[mask].mean())


def psnr(clean01, out01, mask=None) -> float:
    m = mse(clean01, out01, mask)
    return PSNR_CAP_DB if m == 0 else float(10.0 * np.log10(1.0 / m))


def mae(clean01, out01, mask=None) -> float:
    d = np.abs(q01(clean01) - q01(out01))
    return float(d.mean() if mask is None else d[mask].mean())


def ssim(clean01, out01) -> float:
    """Gaussian window 11, sigma 1.5, data range 1, population covariance.
    scikit-image trims a border of (win_size-1)//2 pixels internally."""
    return float(structural_similarity(
        q01(clean01), q01(out01), data_range=SSIM_DATA_RANGE, gaussian_weights=True,
        sigma=SSIM_SIGMA, win_size=SSIM_WIN_SIZE, use_sample_covariance=False))
