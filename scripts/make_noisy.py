"""Generate the deterministic noisy realizations used by EVERY method (Person A).

    python -m scripts.make_noisy [--with-val]

Writes artifacts/noisy/sigma{S}/{image_id}.png for all BSD68 (calibration + test) images. Noisy
values are integer 8-bit levels, so the PNGs are exact. Pre-clipping mean/std are checked.
--with-val also writes the fixed validation realizations to artifacts/val_noisy/ (reference only;
the trainer regenerates them in memory with the same seeds).
"""
from __future__ import annotations

import argparse

import numpy as np

from common import paths
from common.io import load_gray01, load_png8, save_png8
from common.manifest import select
from common.noise import check_noise_stats, make_noisy
from common.protocol import SIGMAS_255, TEST_DATASET, TRAIN_DATASET
from scripts._common import clean_path, get_manifest, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-val", action="store_true")
    a = ap.parse_args()
    m = get_manifest()
    report: dict = {}
    jobs = [(TEST_DATASET, "noisy", None)]
    if a.with_val:
        jobs.append((TRAIN_DATASET, "val_noisy", "validation"))
    for dataset, folder, split in jobs:
        recs = [r for r in select(m, dataset) if (r["split"] == split if split else True)]
        for s in SIGMAS_255:
            stats = []
            for r in recs:
                clean = load_gray01(clean_path(r))
                noisy, noise = make_noisy(clean, dataset, r["index"], s)
                check_noise_stats(noise, s)
                out = paths.artifact_path(f"{folder}/sigma{s}/{r['image_id']}.png")
                save_png8(out, noisy)
                assert np.array_equal(load_png8(out), noisy), "PNG round trip is not exact"
                stats.append((float(noise.mean()), float(noise.std())))
            arr = np.array(stats)
            report[f"{folder}/sigma{s}"] = {"n": len(recs), "preclip_mean_avg": float(arr[:, 0].mean()),
                                            "preclip_std_avg": float(arr[:, 1].mean())}
            print(folder, s, report[f"{folder}/sigma{s}"])
    write_json(paths.artifact_path("logs/noise_check.json"), report)


if __name__ == "__main__":
    main()
