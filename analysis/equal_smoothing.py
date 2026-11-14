"""Equal-smoothing analysis (Person B). Descriptive, pre-declared grid, tunes nothing.

    python -m analysis.equal_smoothing [--methods gaussian bilateral tv nlm] [--sigmas 15 25 50]

For each classical method, sigma and setting of its main smoothing parameter (configs/equal_smoothing.yaml,
other parameters at the frozen calibrated values), compute over the 60 test images the smooth-region
PSNR, edge-region PSNR and edge-region gradient error. Outputs:
  artifacts/results/equal_smoothing.csv            (means per setting)
  artifacts/results/equal_smoothing_per_image.csv  (per image, for bootstrap CIs)
Helpers below read a curve at matched smooth-region PSNR (linear interpolation, NO extrapolation,
rising branch only: smooth-region PSNR peaks and then falls with over-smoothing) and bootstrap that
reading over images.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import metrics as M
from common import paths
from common import regions as R
from common.io import load_gray01, load_png8
from common.manifest import select
from common.protocol import BOOTSTRAP_ALPHA, BOOTSTRAP_N, CLASSICAL_METHODS, GLOBAL_SEED, SIGMAS_255, TEST_DATASET
from common.regions import make_masks
from common.registry import get_module
from scripts._common import clean_path, get_manifest, load_yaml, noisy_path, read_json


def interp_at(x, y, x0: float) -> float:
    """Value of the curve y(x) at x0 by linear interpolation; NaN outside the curve's x-range."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    o = np.argsort(x)
    x, y = x[o], y[o]
    if x0 < x[0] or x0 > x[-1]:
        return float("nan")
    return float(np.interp(x0, x, y))


def rising_branch(per_image: pd.DataFrame, x_col: str = "smooth_psnr") -> list:
    """Setting values from the weakest smoothing up to the peak of mean smooth-region PSNR.
    Beyond the peak, over-smoothing lowers smooth-region PSNR again, so the curve folds back and
    'matched smooth PSNR' would be ambiguous. Matching is therefore done on the rising branch only
    (fixed from the full-data means, so bootstrap resamples use the same branch)."""
    g = per_image.groupby("setting_value")[x_col].mean().sort_index()
    k = int(np.argmax(g.values))
    return list(g.index[:k + 1])


def curve_means(per_image: pd.DataFrame, x_col: str, y_col: str, images=None, branch: bool = False):
    d = per_image if images is None else per_image[per_image["image_id"].isin(images)]
    if branch:
        d = d[d["setting_value"].isin(rising_branch(per_image, x_col))]
    g = d.groupby("setting_value")[[x_col, y_col]].mean().sort_index()
    return g[x_col].values, g[y_col].values


def bootstrap_matched(per_image: pd.DataFrame, target_smooth_psnr: float, y_col: str = "edge_psnr",
                      n_boot: int = BOOTSTRAP_N, alpha: float = BOOTSTRAP_ALPHA, seed: int = GLOBAL_SEED):
    """per_image: rows of ONE method and ONE sigma. Resample images, rebuild the mean curve (rising
    branch), read y at the fixed target smooth-region PSNR. Returns (point, lo, hi); NaN if the
    target is off the curve."""
    keep = rising_branch(per_image)
    d = per_image[per_image["setting_value"].isin(keep)]
    ids = d["image_id"].unique()
    point = interp_at(*curve_means(d, "smooth_psnr", y_col), target_smooth_psnr)
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    piv = d.pivot(index="image_id", columns="setting_value", values=["smooth_psnr", y_col])
    xs, ys = piv["smooth_psnr"].values, piv[y_col].values
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        v = interp_at(xs[idx].mean(0), ys[idx].mean(0), target_smooth_psnr)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 0.9 * n_boot:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def network_points(results: pd.DataFrame) -> pd.DataFrame:
    """One point per (network, loss variant, sigma): image-mean smooth/edge metrics, seeds averaged,
    plus the spread across train seeds (for error bars)."""
    d = results[results["method"].isin(["dncnn", "unet"])]
    per_seed = d.groupby(["method", "loss_variant", "sigma_255", "train_seed"])[
        ["smooth_psnr", "edge_psnr", "gradient_error_edge"]].mean().reset_index()
    agg = per_seed.groupby(["method", "loss_variant", "sigma_255"]).agg(
        smooth_psnr=("smooth_psnr", "mean"), edge_psnr=("edge_psnr", "mean"),
        gradient_error_edge=("gradient_error_edge", "mean"),
        edge_psnr_seed_min=("edge_psnr", "min"), edge_psnr_seed_max=("edge_psnr", "max"),
        smooth_psnr_seed_min=("smooth_psnr", "min"), smooth_psnr_seed_max=("smooth_psnr", "max")).reset_index()
    return agg


def run(methods, sigmas) -> None:
    spec = load_yaml("configs/equal_smoothing.yaml")["sweeps"]
    m = get_manifest()
    recs = select(m, TEST_DATASET, "test")
    th = read_json(paths.artifact_path("masks/thresholds.json", must_exist=True))["default"]
    clean = {r["image_id"]: load_gray01(clean_path(r)) for r in recs}
    masks = {i: make_masks(c, th) for i, c in clean.items()}
    per_rows = []
    for meth in methods:
        mod = get_module(meth)
        frozen = read_json(paths.artifact_path(f"tuning/{meth}_frozen.json", must_exist=True))
        for s in sigmas:
            noisy = {r["image_id"]: load_png8(noisy_path(r, s)) for r in recs}
            for val in spec[meth]["values"]:
                cfg = dict(frozen[str(s)]["config"])
                cfg[spec[meth]["param"]] = float(val)
                for i, c in clean.items():
                    out = mod.denoise(noisy[i], cfg)
                    per_rows.append({
                        "sigma_255": s, "method": meth, "setting_name": spec[meth]["param"],
                        "setting_value": float(val), "image_id": i,
                        "smooth_psnr": M.psnr(c, out, masks[i]["smooth"]),
                        "edge_psnr": M.psnr(c, out, masks[i]["edge"]),
                        "gradient_error": R.gradient_error(c, out),
                        "gradient_error_edge": R.gradient_error(c, out, masks[i]["edge"])})
                print(meth, s, val)
    per = pd.DataFrame(per_rows)
    means = per.groupby(["sigma_255", "method", "setting_name", "setting_value"]).agg(
        smooth_psnr_mean=("smooth_psnr", "mean"), edge_psnr_mean=("edge_psnr", "mean"),
        gradient_error_mean=("gradient_error", "mean"),
        gradient_error_edge_mean=("gradient_error_edge", "mean"), n_images=("image_id", "nunique")).reset_index()
    per.to_csv(paths.artifact_path("results/equal_smoothing_per_image.csv"), index=False)
    means.to_csv(paths.artifact_path("results/equal_smoothing.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(CLASSICAL_METHODS), choices=CLASSICAL_METHODS)
    ap.add_argument("--sigmas", nargs="+", type=int, default=list(SIGMAS_255))
    a = ap.parse_args()
    run(a.methods, a.sigmas)
