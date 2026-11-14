"""Torch tests: skipped where torch is missing, run on Kaggle (and locally with torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.dl_infer import restore
from common.losses import SobelGradLoss, total_loss
from models import build_model, count_params, pad_multiple

from tests.conftest import requires_data


# ------------------------------------------------------------------ loss tests (Section 14.5)
def test_sobel_constant_image_is_zero():
    s = SobelGradLoss()
    x = torch.full((2, 1, 16, 16), 0.4)
    assert s(x, x).item() == 0.0
    assert s(x, torch.full_like(x, 0.9)).item() == pytest.approx(0.0, abs=1e-7)   # constants: same (zero) gradients


def test_identical_inputs_zero_loss_both_variants():
    s = SobelGradLoss()
    x = torch.rand(2, 1, 16, 16)
    assert total_loss(x, x, "P", 0.5, s).item() == 0.0 and total_loss(x, x, "G", 0.5, s).item() == 0.0


def test_ramp_response_equals_slope():
    s = SobelGradLoss()
    ramp = (torch.arange(16.0)[None, :].repeat(16, 1) / 255.0)[None, None]
    gx, gy = s.grads(ramp)
    assert torch.allclose(gx[..., 2:-2, 2:-2], torch.full_like(gx[..., 2:-2, 2:-2], 1 / 255.0), atol=1e-7)
    assert torch.allclose(gy, torch.zeros_like(gy), atol=1e-7)
    scaled = ramp * 3
    assert torch.allclose(s.grads(scaled)[0], 3 * gx, atol=1e-7)


def test_lambda_zero_equals_P_value_and_gradient():
    s = SobelGradLoss()
    pred = torch.rand(2, 1, 16, 16, requires_grad=True)
    tgt = torch.rand(2, 1, 16, 16)
    lg = total_loss(pred, tgt, "G", 0.0, s)
    lp = total_loss(pred, tgt, "P", 0.0, s)
    assert lg.item() == pytest.approx(lp.item())
    g1, = torch.autograd.grad(lg, pred)
    g2, = torch.autograd.grad(lp, pred)
    assert torch.allclose(g1, g2)


def test_symmetry_finite_grads_and_odd_sizes():
    s = SobelGradLoss()
    a, b = torch.rand(1, 1, 17, 23, requires_grad=True), torch.rand(1, 1, 17, 23)
    assert s(a, b).item() == pytest.approx(s(b, a).item())
    loss = total_loss(a, b, "G", 1.0, s)
    loss.backward()
    assert torch.isfinite(a.grad).all()
    with pytest.raises(ValueError):
        total_loss(a, b, "X", 1.0, s)


def test_torch_sobel_matches_numpy_operator():
    from common.regions import sobel_grad_mag  # noqa: F401
    from scipy import ndimage as ndi
    rng = np.random.default_rng(0)
    x = rng.random((20, 24))
    s = SobelGradLoss()
    gx, gy = s.grads(torch.from_numpy(x)[None, None].float())
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8.0
    assert np.allclose(gx[0, 0].numpy(), ndi.correlate(x, kx, mode="nearest"), atol=1e-5)
    assert np.allclose(gy[0, 0].numpy(), ndi.correlate(x, kx.T, mode="nearest"), atol=1e-5)


# ------------------------------------------------------------------ models
def test_param_counts_same_range_and_shapes():
    dn, un = build_model("dncnn"), build_model("unet")
    nd, nu = count_params(dn), count_params(un)
    assert 0.4e6 < nd < 0.7e6 and 0.4e6 < nu < 0.7e6 and 0.7 < nu / nd < 1.3
    x = torch.rand(1, 1, 64, 64)
    assert dn(x).shape == x.shape and un(x).shape == x.shape


@pytest.mark.parametrize("hw", [(321, 481), (481, 321), (180, 180), (64, 64)])
def test_unet_wrapper_handles_any_size(hw):
    m = build_model("unet").eval()
    x = torch.rand(1, 1, *hw)
    with torch.no_grad():
        out = restore(m, x, pad_multiple("unet"))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1


def test_models_are_blind():
    import inspect
    for arch in ("dncnn", "unet"):
        params = inspect.signature(build_model(arch).forward).parameters
        assert list(params) == ["x"], "forward must take only the image"
    from methods.dl_common import denoise_dl
    with pytest.raises(AssertionError):
        denoise_dl(np.zeros((8, 8), np.float32), {"arch": "unet", "checkpoint": "x", "sigma_255": 25})


# ------------------------------------------------------------------ trainer (needs BSD data)
@requires_data
def test_pairing_fingerprint_and_resume(tmp_path):
    from common import paths
    from common.dl_train import stream_fingerprint, train
    from scripts._common import get_manifest

    assert stream_fingerprint(0, (380, 1, 180, 180), 4) == stream_fingerprint(0, (380, 1, 180, 180), 4)
    assert stream_fingerprint(0, (380, 1, 180, 180), 4) != stream_fingerprint(1, (380, 1, 180, 180), 4)
    cfg = {"steps": 4, "batch_size": 2, "lr": 1e-3, "lr_milestones": [0.5], "lr_gamma": 0.5, "val_every": 2}
    kw = dict(manifest=get_manifest(), data_root=paths.data_root(), device="cpu", arch_cfg={"depth": 3, "width": 8})
    full = train("dncnn", "G", 0, cfg, 0.5, tmp_path / "full", **kw)
    paused = train("dncnn", "G", 0, cfg, 0.5, tmp_path / "res", max_hours=1e-9, **kw)
    assert paused["status"] == "paused" and paused["step"] == 2
    done = train("dncnn", "G", 0, cfg, 0.5, tmp_path / "res", **kw)
    assert done["status"] == "done" and done["step"] == 4
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "res" / "last.pt", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-6), k
    assert full["params"] == done["params"]
    pa = train("dncnn", "P", 0, cfg, 0.0, tmp_path / "p", **kw)       # P and G share init + data stream
    import json
    fp = lambda d: json.load(open(tmp_path / d / "meta.json"))["data_stream_fingerprint"]   # noqa: E731
    assert fp("p") == fp("full") and pa["status"] == "done"
