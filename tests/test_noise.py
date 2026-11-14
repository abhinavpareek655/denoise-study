import numpy as np
import pytest

from common.noise import add_awgn_8bit, check_noise_stats, make_noisy, noise_rng


def test_deterministic_and_on_8bit_grid(synth):
    a, _ = make_noisy(synth, "BSD68", 3, 25)
    b, _ = make_noisy(synth, "BSD68", 3, 25)
    assert np.array_equal(a, b) and a.dtype == np.float32
    assert np.allclose(a * 255, np.rint(a * 255))
    assert a.min() >= 0 and a.max() <= 1


def test_different_seeds_differ(synth):
    a, _ = make_noisy(synth, "BSD68", 3, 25)
    assert not np.array_equal(a, make_noisy(synth, "BSD68", 4, 25)[0])
    assert not np.array_equal(a, make_noisy(synth, "BSD68", 3, 15)[0])
    assert not np.array_equal(a, make_noisy(synth, "BSD400", 3, 25)[0])


@pytest.mark.parametrize("sigma", [15, 25, 50])
def test_preclip_stats(synth, sigma):
    big = np.tile(synth, (4, 4))
    _, noise = add_awgn_8bit(big, sigma, noise_rng("BSD68", 0, sigma))
    check_noise_stats(noise, sigma)


def test_torch_sampler_matches_rule():
    torch = pytest.importorskip("torch")
    from common.noise import add_awgn_torch
    clean = torch.round(torch.rand(8, 1, 64, 64) * 255) / 255
    g = torch.Generator().manual_seed(0)
    noisy, sigma = add_awgn_torch(clean, g)
    assert torch.allclose(noisy * 255, torch.round(noisy * 255)) and noisy.min() >= 0 and noisy.max() <= 1
    assert (sigma >= 15).all() and (sigma <= 50).all()
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(add_awgn_torch(clean, g2)[0], noisy)
