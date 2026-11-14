import numpy as np
import pytest

from common.evaluator import input_unchanged
from methods import bilateral, gaussian, nlm, tv

CASES = [(gaussian, {"sigma_px": 1.0}), (bilateral, {"sigma_s_px": 1.5, "sigma_r_255": 40.0}),
         (tv, {"weight": 0.1}), (nlm, {"h": 0.08, "sigma_255": 25.0})]


@pytest.mark.parametrize("mod,cfg", CASES, ids=[c[0].METHOD for c in CASES])
def test_interface(mod, cfg, synth):
    rng = np.random.default_rng(0)
    noisy = np.clip(synth + rng.normal(0, 0.1, synth.shape), 0, 1).astype(np.float32)
    out = input_unchanged(mod.denoise, noisy, cfg)
    assert out.dtype == np.float32 and out.shape == noisy.shape
    assert np.array_equal(out, mod.denoise(noisy, cfg))                  # deterministic
    assert np.abs(out - synth).mean() < np.abs(noisy - synth).mean()     # actually denoises


def test_bilateral_unit_conversion(synth):
    """sigma_r is stored on the 0..255 scale: 255 -> range sigma of 1.0 on [0,1] data."""
    wide = bilateral.denoise(synth, {"sigma_s_px": 1.0, "sigma_r_255": 255.0})
    narrow = bilateral.denoise(synth, {"sigma_s_px": 1.0, "sigma_r_255": 1.0})
    assert np.abs(narrow - synth).max() < 1e-3          # tiny range sigma keeps the edge untouched
    assert np.abs(wide - synth).max() > np.abs(narrow - synth).max()


def test_rejects_bad_input(synth):
    with pytest.raises(AssertionError):
        gaussian.denoise(synth.astype(np.float64), {"sigma_px": 1.0})
