import numpy as np

from common import regions as R
from common.protocol import SENSITIVITY_PCTS


def test_masks_edge_smooth_and_disjoint(synth):
    th = R.fit_thresholds([synth], SENSITIVITY_PCTS["default"])
    mk = R.make_masks(synth, th)
    assert mk["edge"].dtype == bool and mk["edge"].shape == synth.shape
    assert mk["edge"][:, 62:66].any()                       # the step edge is detected
    assert mk["smooth"][:30, :30].all()                     # flat corner is smooth
    assert not (mk["smooth"] & mk["edge"]).any()
    assert not (mk["texture"] & mk["edge"]).any()


def test_thresholds_move_with_percentiles(synth):
    lo = R.fit_thresholds([synth], SENSITIVITY_PCTS["low"])
    hi = R.fit_thresholds([synth], SENSITIVITY_PCTS["high"])
    assert lo["std_smooth"] <= hi["std_smooth"] and lo["grad_edge"] <= hi["grad_edge"]


def test_gradient_error_zero_and_blur_positive(synth):
    from scipy import ndimage as ndi
    assert R.gradient_error(synth, synth) == 0.0
    assert R.gradient_error_sobel(synth, synth) == 0.0
    blur = ndi.gaussian_filter(synth, 2.0).astype(np.float32)
    assert R.gradient_error(synth, blur) > 0 and R.gradient_error_sobel(synth, blur) > 0


def test_sobel_ramp_value():
    x = np.tile(np.arange(20, dtype=np.float64)[None, :] / 255.0, (20, 1))
    g = R.sobel_grad_mag(x)
    assert np.allclose(g[5:-5, 5:-5], 1.0 / 255.0)          # kernels /8 -> response equals the slope
