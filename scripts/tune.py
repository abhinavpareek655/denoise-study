"""Per-sigma tuning of a classical method on the CALIBRATION images only.

    python -m scripts.tune --method bilateral

Reads configs/grids/{method}.yaml (declared in advance), evaluates every grid point on the 8
calibration images at each sigma (objective: mean quantized PSNR; ties -> first in grid), writes
artifacts/tuning/{method}_sigma{S}_grid.csv and artifacts/tuning/{method}_frozen.json.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common import metrics as M
from common import paths
from common.io import load_gray01, load_png8
from common.manifest import select
from common.protocol import SIGMAS_255, TEST_DATASET
from common.registry import config_id, get_module
from scripts._common import clean_path, get_manifest, load_yaml, noisy_path, write_json


def grid_points(spec: dict, sigma: int) -> list[dict]:
    g = spec["grid"].get(sigma, spec["grid"].get(str(sigma), spec["grid"].get("all")))
    keys = list(g)
    pts = []
    for vals in itertools.product(*[g[k] for k in keys]):
        cfg = dict(spec.get("fixed", {}))
        cfg.update({k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                    for k, v in zip(keys, vals)})
        if spec.get("inject_sigma"):
            cfg["sigma_255"] = float(sigma)
        pts.append(cfg)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--grid", default=None)
    a = ap.parse_args()
    mod = get_module(a.method)
    spec = load_yaml(a.grid or f"configs/grids/{a.method}.yaml")
    m = get_manifest()
    calib = select(m, TEST_DATASET, "calibration")
    clean = {r["image_id"]: load_gray01(clean_path(r)) for r in calib}
    frozen: dict = {}
    for s in SIGMAS_255:
        noisy = {r["image_id"]: load_png8(noisy_path(r, s)) for r in calib}
        rows = []
        for cfg in grid_points(spec, s):
            scores = [M.psnr(clean[i], mod.denoise(noisy[i], cfg)) for i in clean]
            rows.append({"parameter_config": config_id(cfg), "mean_psnr": float(np.mean(scores)),
                         "config": cfg})
        df = pd.DataFrame(rows)
        df.drop(columns="config").to_csv(paths.artifact_path(f"tuning/{a.method}_sigma{s}_grid.csv"), index=False)
        best = int(df["mean_psnr"].values.argmax())          # first maximum = grid-order tie-break
        frozen[str(s)] = {"config": rows[best]["config"], "parameter_config": rows[best]["parameter_config"],
                          "calibration_mean_psnr": rows[best]["mean_psnr"]}
        print(f"{a.method} sigma={s}: best {rows[best]['parameter_config']} -> {rows[best]['mean_psnr']:.2f} dB")
    frozen["_meta"] = {"method": a.method, "tuned_on": [r["image_id"] for r in calib],
                       "objective": "mean quantized PSNR on calibration images", "manifest_images_sha256": m["images_sha256"]}
    write_json(paths.artifact_path(f"tuning/{a.method}_frozen.json"), frozen)


if __name__ == "__main__":
    main()
