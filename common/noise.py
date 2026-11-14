"""Noise generation. Noisy-image rule (identical in training and test):

    noisy = clip(round(255*clean + N(0, sigma^2)), 0, 255) / 255
"""
from __future__ import annotations

import numpy as np

from common.protocol import DATASET_CODES, GLOBAL_SEED, TRAIN_SIGMA_RANGE_255


def noise_rng(dataset: str, image_index: int, sigma_255: int, global_seed: int = GLOBAL_SEED):
    """Deterministic generator per (dataset, image, sigma). Never use Python hash()."""
    ss = np.random.SeedSequence([global_seed, DATASET_CODES[dataset], int(image_index), int(sigma_255)])
    return np.random.default_rng(ss)


def add_awgn_8bit(clean01: np.ndarray, sigma_255: float, rng: np.random.Generator):
    """Apply the noisy-image rule. Returns (noisy01 float32, pre-clip noise float64)."""
    clean255 = np.rint(clean01.astype(np.float64) * 255.0)   # clean already on the 8-bit grid
    noise = rng.normal(0.0, float(sigma_255), size=clean255.shape)
    noisy255 = np.clip(np.rint(clean255 + noise), 0.0, 255.0)
    return (noisy255 / 255.0).astype(np.float32), noise


def check_noise_stats(noise: np.ndarray, sigma_255: float, k: float = 4.0) -> None:
    """Pre-clipping mean ~ 0 and std ~ sigma (tolerances from sampling error)."""
    se_mean = sigma_255 / np.sqrt(noise.size)
    assert abs(noise.mean()) < k * se_mean, f"pre-clip noise mean {noise.mean():.4f} off"
    assert abs(noise.std() / sigma_255 - 1.0) < 0.02, f"pre-clip noise std {noise.std():.4f} off"


def make_noisy(clean01: np.ndarray, dataset: str, image_index: int, sigma_255: int):
    """Deterministic noisy realization for a manifest image. Returns (noisy01, noise)."""
    return add_awgn_8bit(clean01, sigma_255, noise_rng(dataset, image_index, sigma_255))


def add_awgn_torch(clean01, gen, sigma_range=TRAIN_SIGMA_RANGE_255):
    """Training noise: sigma ~ U[lo, hi] per patch, same rule as test noise.

    clean01: (B,1,H,W) tensor on the 8-bit grid. `gen`: CPU torch.Generator (deterministic).
    Returns (noisy01 on clean01's device, sigma_255 (B,1,1,1)). sigma is for logging only and is
    NEVER fed to the model (blind training).
    """
    import torch

    b = clean01.shape[0]
    lo, hi = sigma_range
    sigma = lo + (hi - lo) * torch.rand(b, 1, 1, 1, generator=gen)
    noise = torch.randn(tuple(clean01.shape), generator=gen) * sigma
    clean255 = torch.round(clean01.detach().cpu() * 255.0)
    noisy = torch.clamp(torch.round(clean255 + noise), 0.0, 255.0) / 255.0
    return noisy.to(clean01.device), sigma
