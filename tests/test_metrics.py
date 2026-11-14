import numpy as np
from scipy import ndimage as ndi

from common import metrics as M
from common.protocol import PSNR_CAP_DB


def test_psnr_identical_is_capped(synth):
    assert M.psnr(synth, synth) == PSNR_CAP_DB


def test_psnr_known_mse(synth):
    out = np.clip(synth + 10 / 255, 0, 1).astype(np.float32)
    out[synth > 0.95] = synth[synth > 0.95]
    m = M.mse(synth, out)
    assert abs(M.psnr(synth, out) - 10 * np.log10(1 / m)) < 1e-9
    assert M.psnr(synth, out) > 25


def test_quantization_applied():
    a = np.full((20, 20), 0.5, dtype=np.float32)
    b = a + 0.1 / 255            # sub-level change vanishes after 8-bit rounding
    assert M.mse(a, b) == 0


def test_ssim_identical_is_one(synth):
    assert abs(M.ssim(synth, synth) - 1.0) < 1e-9


def _ssim_reference(x, y, sigma=1.5, k1=0.01, k2=0.03):
    """Independent Gaussian-window SSIM (window 11 = truncate 3.5 at sigma 1.5), border cropped by 5."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    g = lambda a: ndi.gaussian_filter(a, sigma, truncate=3.5, mode="reflect")      # noqa: E731
    mx, my = g(x), g(y)
    vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    c1, c2 = (k1) ** 2, (k2) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(s[5:-5, 5:-5].mean())


def test_ssim_matches_independent_reference(synth):
    rng = np.random.default_rng(0)
    noisy = np.clip(np.rint((synth + rng.normal(0, 25 / 255, synth.shape)) * 255), 0, 255) / 255
    noisy = noisy.astype(np.float32)
    assert abs(M.ssim(synth, noisy) - _ssim_reference(M.q01(synth), M.q01(noisy))) < 2e-3
