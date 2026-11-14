"""Score every method output against the clean image and write the per-image results CSV.

    python -m scripts.evaluate [--method-ids gaussian unet_P_s0 ...] [--mask-variant default|low|high] [--strict]

Reads artifacts/outputs/*/run_meta.json, noisy PNGs (baseline `noisy` row), thresholds.json, and
results/runtime_*.csv. Replaces rows of the evaluated method ids and keeps the rest, so A and B can
evaluate their own methods separately and merge the CSVs. Integrity problems are printed (and fail
the run with --strict). Test split only.
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from common import paths
from common.evaluator import integrity_check, order_columns, score_image
from common.io import load_gray01, load_png8
from common.manifest import select
from common.protocol import GLOBAL_SEED, SENSITIVITY_PCTS, SIGMAS_255, TEST_DATASET
from common.regions import make_masks
from scripts._common import clean_path, get_manifest, noisy_path, read_json


def _rows_for(method_id, meta, recs, clean, masks, runtime):
    rows = []
    for s in SIGMAS_255:
        for r in recs:
            iid = r["image_id"]
            if meta is None:     # noisy baseline
                out, pcfg, method, lv, ts, mid = load_png8(noisy_path(r, s)), "none", "noisy", "none", None, ""
                hw = ""
            else:
                f = paths.artifact_path(f"outputs/{method_id}/sigma{s}/{iid}.png", must_exist=True)
                out, pcfg = load_png8(f), meta["parameter_config"][str(s)]
                method, lv, ts = meta["method"], meta["loss_variant"], meta["train_seed"]
                mid = method_id if method in ("dncnn", "unet") else ""
                hw = meta["hardware"]
            fields, warns = score_image(clean[iid], out, masks[iid])
            rt = runtime.get((iid, s), (math.nan, hw))
            rows.append({"image_id": iid, "split": r["split"], "seed": GLOBAL_SEED, "sigma_255": s,
                         "method": method, "parameter_config": pcfg, "model_id_optional": mid,
                         "loss_variant": lv, "train_seed": ts, **fields, "runtime_seconds": rt[0],
                         "hardware": rt[1] or hw, "warnings": ";".join(warns)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-ids", nargs="*", default=None)
    ap.add_argument("--mask-variant", default="default", choices=list(SENSITIVITY_PCTS))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    m = get_manifest()
    recs = select(m, TEST_DATASET, "test")
    th = read_json(paths.artifact_path("masks/thresholds.json", must_exist=True))[a.mask_variant]
    clean = {r["image_id"]: load_gray01(clean_path(r)) for r in recs}
    masks = {i: make_masks(c, th) for i, c in clean.items()}
    out_root = paths.artifact_path("outputs")
    ids = a.method_ids or sorted(p.name for p in out_root.iterdir() if (p / "run_meta.json").exists())
    rows = _rows_for("noisy", None, recs, clean, masks, {}) if not a.method_ids or "noisy" in a.method_ids else []
    for mid in [i for i in ids if i != "noisy"]:
        meta = read_json(out_root / mid / "run_meta.json")
        rt_path = paths.artifact_path(f"results/runtime_{mid}.csv")
        runtime = {}
        if rt_path.exists():
            d = pd.read_csv(rt_path)
            runtime = {(r.image_id, int(r.sigma_255)): (r.runtime_seconds, r.hardware) for r in d.itertuples()}
        rows += _rows_for(mid, meta, recs, clean, masks, runtime)
        print("scored", mid)
    new = order_columns(pd.DataFrame(rows))
    name = "results.csv" if a.mask_variant == "default" else f"results_{a.mask_variant}.csv"
    p = paths.artifact_path(f"results/{name}")
    if p.exists():
        old = pd.read_csv(p)
        done = {(r.method, r.model_id_optional if isinstance(r.model_id_optional, str) else "") for r in new.itertuples()}
        mask = [(r.method, r.model_id_optional if isinstance(r.model_id_optional, str) else "") not in done
                for r in old.itertuples()]
        old = old[mask]
        new = pd.concat([old, new], ignore_index=True) if len(old) else new
    problems = integrity_check(new)
    p.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(p, index=False)
    print(f"wrote {p}: {len(new)} rows; manifest digest {m['images_sha256'][:12]}")
    for pr in problems:
        print("INTEGRITY:", pr)
    if problems and a.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
