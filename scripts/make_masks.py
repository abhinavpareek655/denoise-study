"""Fit region thresholds on the CALIBRATION images only and report pixel counts (Person B).

    python -m scripts.make_masks

Writes artifacts/masks/thresholds.json (default + sensitivity variants) and
artifacts/masks/pixel_counts.csv (every BSD68 image x variant). Masks themselves are recomputed on
the fly from the clean image and these thresholds (they are cheap and deterministic).
"""
from __future__ import annotations

import pandas as pd

from common import paths
from common.io import load_gray01
from common.manifest import select
from common.protocol import SENSITIVITY_PCTS, TEST_DATASET
from common.regions import fit_thresholds, make_masks
from scripts._common import clean_path, get_manifest, write_json


def main() -> None:
    m = get_manifest()
    calib = select(m, TEST_DATASET, "calibration")
    clean_cal = [load_gray01(clean_path(r)) for r in calib]
    th = {name: fit_thresholds(clean_cal, pcts) for name, pcts in SENSITIVITY_PCTS.items()}
    th["fitted_on"] = [r["image_id"] for r in calib]
    write_json(paths.artifact_path("masks/thresholds.json"), th)
    rows = []
    for r in select(m, TEST_DATASET):
        c = load_gray01(clean_path(r))
        for name in SENSITIVITY_PCTS:
            mk = make_masks(c, th[name])
            rows.append({"image_id": r["image_id"], "split": r["split"], "variant": name,
                         "n_pixels": c.size, **{f"n_{k}": int(v.sum()) for k, v in mk.items()},
                         "overlap_smooth_edge": int((mk["smooth"] & mk["edge"]).sum())})
    df = pd.DataFrame(rows)
    out = paths.artifact_path("masks/pixel_counts.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("variant")[["n_smooth", "n_edge", "n_texture"]].mean().round(0))


if __name__ == "__main__":
    main()
