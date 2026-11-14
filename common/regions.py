"""Region masks (built from the CLEAN image) and gradient error.

Headline gradient operator: Gaussian derivative (sigma 1 px). The Sobel variant is reported only
as "same operator as the G loss" because G is trained on Sobel gradients (circular by design).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from common.metrics import q01
from common.protocol import GRAD_SIGMA_PX, LOCAL_STD_SIZE


def grad_mag_gauss(img01: np.ndarray, sigma_px: float = GRAD_SIGMA_PX) -> np.ndarray:
    return ndi.gaussian_gradient_magnitude(img01.astype(np.float64), sigma=sigma_px, mode="reflect")


def local_std(img01: np.ndarray, size: int = LOCAL_STD_SIZE) -> np.ndarray:
    x = img01.astype(np.float64)
    m = ndi.uniform_filter(x, size=size, mode="reflect")
    m2 = ndi.uniform_filter(x * x, size=size, mode="reflect")
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def fit_thresholds(clean_calib: list[np.ndarray], pcts: dict) -> dict:
    """Percentiles pooled over ALL pixels of the calibration images. Never fit on test images."""
    g = np.concatenate([grad_mag_gauss(c).ravel() for c in clean_calib])
    s = np.concatenate([local_std(c).ravel() for c in clean_calib])
    return {
        "grad_smooth": float(np.percentile(g, pcts["smooth"])),
        "std_smooth": float(np.percentile(s, pcts["smooth"])),
        "grad_edge": float(np.percentile(g, pcts["edge"])),
        "std_texture": float(np.percentile(s, pcts["texture"])),
        "pcts": dict(pcts),
    }


def make_masks(clean01: np.ndarray, th: dict) -> dict:
    g, s = grad_mag_gauss(clean01), local_std(clean01)
    edge = g >= th["grad_edge"]
    smooth = (g <= th["grad_smooth"]) & (s <= th["std_smooth"])
    texture = (s >= th["std_texture"]) & ~edge
    return {"smooth": smooth, "edge": edge, "texture": texture}


def gradient_error(clean01, out01, mask=None) -> float:
    d = np.abs(grad_mag_gauss(q01(clean01)) - grad_mag_gauss(q01(out01)))
    return float(d.mean() if mask is None else d[mask].mean())


def sobel_grad_mag(img01: np.ndarray) -> np.ndarray:
    """Sobel kernels divided by 8; border = replicate (matches the torch loss)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    x = img01.astype(np.float64)
    return np.hypot(ndi.correlate(x, kx, mode="nearest"), ndi.correlate(x, kx.T, mode="nearest"))


def gradient_error_sobel(clean01, out01, mask=None) -> float:
    """'Same operator as the G loss'. Never a headline number."""
    d = np.abs(sobel_grad_mag(q01(clean01)) - sobel_grad_mag(q01(out01)))
    return float(d.mean() if mask is None else d[mask].mean())
