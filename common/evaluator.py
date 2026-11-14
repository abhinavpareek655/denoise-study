"""Per-image scoring and integrity checks for the results CSV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import metrics as M
from common import regions as R
from common.io import array_hash, check_image
from common.protocol import CSV_COLUMNS, CSV_EXTRA_COLUMNS, MIN_MASK_PIXELS, PRIMARY_KEY

NAN = float("nan")


def score_image(clean01: np.ndarray, out01: np.ndarray, masks: dict) -> tuple[dict, list[str]]:
    """All metric fields for one (clean, output) pair. Returns (fields, warnings)."""
    warns: list[str] = []
    assert clean01.shape == out01.shape, f"shape mismatch {clean01.shape} vs {out01.shape}"
    assert np.isfinite(out01).all(), "non-finite output"
    f: dict = {"psnr": M.psnr(clean01, out01), "ssim": M.ssim(clean01, out01),
               "gradient_error": R.gradient_error(clean01, out01),
               "gradient_error_sobel": R.gradient_error_sobel(clean01, out01)}
    if M.mse(clean01, out01) == 0:
        warns.append("psnr_capped")
    for name in ("smooth", "edge", "texture"):
        n = int(masks[name].sum())
        f[f"n_{name}_px"] = n
        if n < MIN_MASK_PIXELS:
            warns.append(f"small_mask_{name}_{n}")
            f[f"{name}_psnr"] = NAN
            f[f"{name}_mae"] = NAN
        else:
            f[f"{name}_psnr"] = M.psnr(clean01, out01, masks[name])
            f[f"{name}_mae"] = M.mae(clean01, out01, masks[name])
    if f["n_edge_px"] >= MIN_MASK_PIXELS:
        f["gradient_error_edge"] = R.gradient_error(clean01, out01, masks["edge"])
        f["gradient_error_sobel_edge"] = R.gradient_error_sobel(clean01, out01, masks["edge"])
    else:
        f["gradient_error_edge"] = f["gradient_error_sobel_edge"] = NAN
    return f, warns


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_COLUMNS + CSV_EXTRA_COLUMNS)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in CSV_COLUMNS + CSV_EXTRA_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CSV_COLUMNS + CSV_EXTRA_COLUMNS]


def integrity_check(df: pd.DataFrame, expected_rows: int | None = None) -> list[str]:
    """Return a list of problems (empty = clean). Callers decide whether to raise."""
    problems: list[str] = []
    key = df[PRIMARY_KEY].fillna("").astype(str).apply("|".join, axis=1)
    dups = key[key.duplicated()]
    if len(dups):
        problems.append(f"{len(dups)} duplicate primary keys, e.g. {dups.iloc[0]}")
    for c in ("psnr", "ssim", "gradient_error", "gradient_error_sobel"):
        if df[c].isna().any():
            problems.append(f"missing values in {c}")
        if not np.isfinite(df[c].astype(float)).all():
            problems.append(f"non-finite values in {c}")
    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"row count {len(df)} != expected {expected_rows}")
    if (df["split"] != "test").any():
        problems.append("non-test rows in the main results table")
    noisy = df[df["method"] == "noisy"]
    for s, ref in ((15, 24.6), (25, 20.2), (50, 14.2)):
        v = noisy[noisy["sigma_255"] == s]["psnr"].astype(float)
        if len(v) and not (ref - 0.5 < v.mean() < ref + 1.5):
            problems.append(f"noisy PSNR at sigma {s} = {v.mean():.2f} dB, theory {ref}")
    return problems


def input_unchanged(fn, image: np.ndarray, *args, **kw) -> np.ndarray:
    """Call fn(image, ...) and assert the input array was not modified in place."""
    before = array_hash(image)
    out = fn(image, *args, **kw)
    assert array_hash(image) == before, "method modified its input in place"
    check_image(out, "method output")
    assert out.shape == image.shape
    return out


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
