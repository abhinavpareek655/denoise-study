"""Run one method on the BSD68 noisy images and store quantized outputs + per-image runtime.

    python -m scripts.infer --method gaussian                       # uses artifacts/tuning/*_frozen.json
    python -m scripts.infer --method unet --model-id unet_P_s0 [--device cuda]
    options: --splits test calibration   --sigmas 15 25 50

Outputs: artifacts/outputs/{method_id}/sigma{S}/{image_id}.png, run_meta.json, and
artifacts/results/runtime_{method_id}.csv. method_id = method (classical) or the DL model id.
Runtime = wall time of one denoise call (GPU synchronised; DL model loading excluded, host->device
transfer included).
"""
from __future__ import annotations

import argparse
import csv
import time

from common import paths
from common.io import load_png8, save_png8
from common.manifest import select
from common.protocol import CLASSICAL_METHODS, DL_METHODS, SIGMAS_255, TEST_DATASET
from common.registry import config_id, get_module
from common.versions import env_info
from scripts._common import get_manifest, noisy_path, pick_device, read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=CLASSICAL_METHODS + DL_METHODS)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--sigmas", nargs="+", type=int, default=list(SIGMAS_255))
    a = ap.parse_args()

    mod = get_module(a.method)
    is_dl = a.method in DL_METHODS
    device = pick_device(a.device) if is_dl else "cpu"
    method_id = a.model_id if is_dl else a.method
    if is_dl and not a.model_id:
        raise SystemExit("DL methods need --model-id, e.g. unet_P_s0")
    meta = read_json(paths.artifact_path(f"checkpoints/{a.model_id}/meta.json", must_exist=True)) if is_dl else {}
    frozen = None if is_dl else read_json(paths.artifact_path(f"tuning/{a.method}_frozen.json", must_exist=True))
    info = env_info(use_gpu=device.startswith("cuda"))
    m = get_manifest()
    recs = [r for sp in a.splits for r in select(m, TEST_DATASET, sp)]
    run_meta = {"method": a.method, "method_id": method_id, "device": device, "hardware": info["hardware"],
                "loss_variant": meta.get("loss_variant", "none"), "train_seed": meta.get("train_seed"),
                "parameter_config": {}, "environment": info}
    rt_rows = []
    for s in a.sigmas:
        if is_dl:
            cfg = {"arch": a.method, "device": device,
                   "checkpoint": str(paths.artifact_path(f"checkpoints/{a.model_id}/best.pt", must_exist=True))}
            pcfg = a.model_id
            fn = lambda x, c=cfg: mod.denoise_dl(x, c)          # noqa: E731
            fn(load_png8(noisy_path(recs[0], s)))               # warm-up (model load, cudnn)
        else:
            cfg = frozen[str(s)]["config"]
            pcfg = frozen[str(s)]["parameter_config"]
            fn = lambda x, c=cfg: mod.denoise(x, c)             # noqa: E731
        run_meta["parameter_config"][str(s)] = pcfg
        for r in recs:
            x = load_png8(noisy_path(r, s))
            t0 = time.perf_counter()
            y = fn(x)
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            save_png8(paths.artifact_path(f"outputs/{method_id}/sigma{s}/{r['image_id']}.png"), y)
            rt_rows.append({"image_id": r["image_id"], "sigma_255": s, "runtime_seconds": dt,
                            "hardware": info["hardware"], "parameter_config": pcfg})
        print(f"{method_id} sigma={s}: {len(recs)} images done")
    write_json(paths.artifact_path(f"outputs/{method_id}/run_meta.json"), run_meta)
    p = paths.artifact_path(f"results/runtime_{method_id}.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rt_rows[0]))
        w.writeheader()
        w.writerows(rt_rows)


if __name__ == "__main__":
    main()
