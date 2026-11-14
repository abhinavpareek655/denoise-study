"""Image-level statistics (Person A). Images are the experimental unit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.protocol import BOOTSTRAP_ALPHA, BOOTSTRAP_N, GLOBAL_SEED


def bootstrap_ci(diffs, n_boot: int = BOOTSTRAP_N, alpha: float = BOOTSTRAP_ALPHA,
                 seed: int = GLOBAL_SEED) -> tuple[float, float, float]:
    """Resample IMAGES with replacement. Returns (mean, lo, hi) percentile CI of the mean."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def per_image(df: pd.DataFrame, method_id: str, metric: str, sigma: int) -> pd.Series:
    """Per-image values for one method id (DL: mean over train seeds). Index = image_id."""
    d = df[(df["sigma_255"] == sigma) & (df["_method_id"] == method_id)]
    return d.groupby("image_id")[metric].mean()


def add_method_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_method_id`: classical -> method; DL -> '{method}_{loss_variant}' (seeds pooled)."""
    out = df.copy()
    dl = out["loss_variant"].isin(["P", "G"])
    out["_method_id"] = np.where(dl, out["method"] + "_" + out["loss_variant"].astype(str), out["method"])
    return out


def paired_diff(df: pd.DataFrame, id_a: str, id_b: str, metric: str, sigma: int) -> pd.Series:
    """Per-image difference a - b on the images both have."""
    d = add_method_id(df)
    a, b = per_image(d, id_a, metric, sigma), per_image(d, id_b, metric, sigma)
    return (a - b).dropna()


def seed_spread(df: pd.DataFrame, method: str, loss_variant: str, metric: str, sigma: int) -> dict:
    """Spread of the image-mean metric across training seeds."""
    d = df[(df["method"] == method) & (df["loss_variant"] == loss_variant) & (df["sigma_255"] == sigma)]
    per_seed = d.groupby("train_seed")[metric].mean()
    return {"per_seed": per_seed.to_dict(), "mean": float(per_seed.mean()),
            "min": float(per_seed.min()), "max": float(per_seed.max()),
            "std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")}
