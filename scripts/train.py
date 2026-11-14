"""Train one DL configuration (resumable). Same code for DnCNN (A) and U-Net (B).

    python -m scripts.train --arch unet --variant P --seed 0
    python -m scripts.train --arch unet --variant G --seed 0            # needs LAMBDA_FROZEN
    python -m scripts.train --arch unet --variant G --lam 0.5 --pilot   # lambda pilot (validation only)
    python -m scripts.train --arch unet --variant P --seed 0 --bench 200  # throughput test

Re-running the same command after a paused/killed session resumes from last.pt.
"""
from __future__ import annotations

import argparse
import csv

from common import paths, protocol as P
from common.dl_train import train
from scripts._common import get_manifest, load_yaml, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=P.DL_METHODS)
    ap.add_argument("--variant", required=True, choices=P.LOSS_VARIANTS)
    ap.add_argument("--seed", type=int, default=0, choices=P.TRAIN_SEEDS)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--config", default="configs/train/train.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pilot", action="store_true", help="short G run for the lambda pilot")
    ap.add_argument("--bench", type=int, default=0, help="run N steps, print steps/s, exit")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    lam = a.lam if a.lam is not None else P.LAMBDA_FROZEN
    if a.variant == "G" and lam is None:
        raise SystemExit("G needs --lam (pilot) or protocol.LAMBDA_FROZEN set after the pilot")
    if a.pilot:
        assert a.variant == "G" and a.lam in P.LAMBDA_GRID, "pilot: --variant G --lam in LAMBDA_GRID"
    steps = a.steps or (cfg["pilot_steps"] if a.pilot else cfg["steps"])
    model_id = (f"{a.arch}_G_pilot_lam{a.lam}_s{a.seed}" if a.pilot else f"{a.arch}_{a.variant}_s{a.seed}")
    out_dir = paths.artifact_path(f"checkpoints/{model_id}")
    res = train(a.arch, a.variant, a.seed, cfg, lam or 0.0, out_dir, get_manifest(), paths.data_root(),
                device=pick_device(a.device), steps=steps, max_hours=a.max_hours or cfg.get("max_hours"),
                arch_cfg=cfg["models"][a.arch], resume=not a.no_resume, bench=a.bench)
    print(res)
    if a.pilot and res["status"] == "done":
        row = {"arch": a.arch, "lambda": a.lam, "seed": a.seed, "steps": steps,
               "val_psnr_mean": res["best"]["psnr_mean"], "val_grad_err_mean": res["best"]["grad_err_mean"]}
        p = paths.artifact_path(f"tuning/lambda_pilot_{a.arch}.csv")
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
