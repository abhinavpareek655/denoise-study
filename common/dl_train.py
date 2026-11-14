"""Shared trainer used by BOTH networks (only the model differs). Resumable, Kaggle-safe.

Pairing guarantee: for a given train_seed, P and G use identical weight init, identical patch
stream (positions, flips, rotations) and identical noise draws, because three independent
generators (init / data / noise) are seeded from train_seed and loss code never touches them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import metrics as M
from common import protocol as P
from common import regions as R
from common.dl_infer import restore
from common.io import load_gray01, sha256_file
from common.losses import SobelGradLoss, total_loss
from common.manifest import select
from common.noise import add_awgn_torch, make_noisy
from common.versions import env_info
from models import build_model, count_params, pad_multiple

TAGS = {"init": 1, "data": 2, "noise": 3}
FP_BATCHES = 20


def derive_seed(train_seed: int, tag: str) -> int:
    ss = np.random.SeedSequence([P.GLOBAL_SEED, 7, int(train_seed), TAGS[tag]])
    return int(ss.generate_state(1)[0])


def make_generators(train_seed: int) -> dict:
    return {k: torch.Generator().manual_seed(derive_seed(train_seed, k)) for k in ("init", "data", "noise")}


# ---------------------------------------------------------------------------- data
def load_train_tensor(manifest: dict, data_root: Path) -> torch.Tensor:
    recs = select(manifest, P.TRAIN_DATASET, "train")
    imgs = [load_gray01(Path(data_root) / r["relative_path"]) for r in recs]
    shapes = {i.shape for i in imgs}
    assert len(shapes) == 1, f"training images must share one size, got {shapes}"
    return torch.from_numpy(np.stack(imgs))[:, None]            # (N,1,H,W) float32 in [0,1]


class PatchSampler:
    """Random 64x64 crops + random flip + random 90-degree rotation, from the `data` generator."""

    def __init__(self, images: torch.Tensor, gen: torch.Generator, patch: int = P.PATCH_SIZE):
        self.images, self.gen, self.patch = images, gen, patch

    def draw(self, batch: int):
        n, _, h, w = self.images.shape
        g, ps = self.gen, self.patch
        idx = torch.randint(0, n, (batch,), generator=g)
        ys = torch.randint(0, h - ps + 1, (batch,), generator=g)
        xs = torch.randint(0, w - ps + 1, (batch,), generator=g)
        flips = torch.randint(0, 2, (batch,), generator=g)
        rots = torch.randint(0, 4, (batch,), generator=g)
        return idx, ys, xs, flips, rots

    def next_batch(self, batch: int) -> torch.Tensor:
        idx, ys, xs, flips, rots = self.draw(batch)
        out = []
        for i in range(batch):
            p = self.images[idx[i], :, ys[i]:ys[i] + self.patch, xs[i]:xs[i] + self.patch]
            if flips[i]:
                p = p.flip(-1)
            out.append(torch.rot90(p, int(rots[i]), (-2, -1)))
        return torch.stack(out)


def stream_fingerprint(train_seed: int, images_shape: tuple, batch: int, n_batches: int = FP_BATCHES) -> str:
    """Hash of the first batches of patch indices; must be identical for P and G at one seed."""
    dummy = torch.zeros(images_shape)
    s = PatchSampler(dummy, make_generators(train_seed)["data"])
    h = hashlib.sha256()
    for _ in range(n_batches):
        for t in s.draw(batch):
            h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------- validation
def load_validation(manifest: dict, data_root: Path) -> dict:
    """The 20 BSD400 validation images with fixed noisy realizations at every test sigma."""
    recs = select(manifest, P.TRAIN_DATASET, "validation")
    clean = [load_gray01(Path(data_root) / r["relative_path"]) for r in recs]
    noisy = {s: [make_noisy(c, P.TRAIN_DATASET, r["index"], s)[0] for c, r in zip(clean, recs)]
             for s in P.SIGMAS_255}
    return {"clean": clean, "noisy": noisy, "ids": [r["image_id"] for r in recs]}


def validate(model, val: dict, arch: str, device: str) -> dict:
    model.eval()
    res: dict = {"psnr_by_sigma": {}, "grad_err_by_sigma": {}}
    with torch.no_grad():
        for s in P.SIGMAS_255:
            x = torch.from_numpy(np.stack(val["noisy"][s]))[:, None].to(device)
            out = restore(model, x, pad_multiple(arch), clip=True).squeeze(1).cpu().numpy()
            res["psnr_by_sigma"][s] = float(np.mean([M.psnr(c, o) for c, o in zip(val["clean"], out)]))
            res["grad_err_by_sigma"][s] = float(np.mean(
                [R.gradient_error(c, o) for c, o in zip(val["clean"], out)]))
    model.train()
    res["psnr_mean"] = float(np.mean(list(res["psnr_by_sigma"].values())))
    res["grad_err_mean"] = float(np.mean(list(res["grad_err_by_sigma"].values())))
    return res


# ---------------------------------------------------------------------------- checkpoints
def save_ckpt(path: Path, model, opt, step, gens, best, log, extra=None) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "gen_init": gens["init"].get_state(), "gen_data": gens["data"].get_state(),
                "gen_noise": gens["noise"].get_state(), "best": best, "log": log, **(extra or {})}, tmp)
    tmp.replace(path)                 # atomic: a killed session never leaves a half-written file


def try_resume(path: Path, model, opt, gens):
    if not Path(path).exists():
        return 0, {"psnr_mean": -1.0, "step": 0}, []
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    for k in ("init", "data", "noise"):
        gens[k].set_state(ck[f"gen_{k}"])
    return int(ck["step"]), ck["best"], ck["log"]


def lr_at(step: int, base_lr: float, steps: int, milestones: list[float], gamma: float) -> float:
    return base_lr * gamma ** sum(step >= int(m * steps) for m in milestones)


# ---------------------------------------------------------------------------- main loop
def train(arch: str, variant: str, train_seed: int, cfg: dict, lam: float, out_dir: Path,
          manifest: dict, data_root: Path, device: str = "cpu", steps: int | None = None,
          max_hours: float | None = None, arch_cfg: dict | None = None, resume: bool = True,
          bench: int = 0) -> dict:
    """Train one (arch, variant, seed). Returns a status dict. Re-run the same call to resume."""
    assert variant in P.LOSS_VARIANTS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(steps if steps is not None else cfg["steps"])
    bs = int(cfg["batch_size"])
    gens = make_generators(train_seed)
    torch.manual_seed(derive_seed(train_seed, "init"))          # same init for P and G
    model = build_model(arch, arch_cfg).to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    sobel = SobelGradLoss().to(device)
    images = load_train_tensor(manifest, data_root)
    sampler = PatchSampler(images, gens["data"])
    val = load_validation(manifest, data_root)
    fp = stream_fingerprint(train_seed, tuple(images.shape), bs)

    last, best_path = out_dir / "last.pt", out_dir / "best.pt"
    step, best, log = (try_resume(last, model, opt, gens) if (resume and not bench)
                       else (0, {"psnr_mean": -1.0, "step": 0}, []))
    model.train()
    t0, status = time.time(), "done"
    val_every = int(cfg["val_every"])
    log_path = out_dir / "train_log.csv"
    run_steps = 0
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, float(cfg["lr"]), steps, list(cfg["lr_milestones"]), float(cfg["lr_gamma"]))
        clean = sampler.next_batch(bs).to(device)
        noisy, _sigma = add_awgn_torch(clean, gens["noise"])      # sigma never reaches the model
        restored = noisy - model(noisy)
        loss = total_loss(restored, clean, variant, lam, sobel)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        run_steps += 1
        if bench and run_steps >= bench:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return {"status": "bench", "steps_per_sec": run_steps / (time.time() - t0), "params": n_params}
        if step % val_every == 0 or step == steps:
            v = validate(model, val, arch, device)
            row = {"step": step, "loss": float(loss.item()), "lr": opt.param_groups[0]["lr"],
                   "val_psnr_mean": v["psnr_mean"], "val_grad_err_mean": v["grad_err_mean"],
                   "elapsed_s": round(time.time() - t0, 1)}
            for s in P.SIGMAS_255:
                row[f"val_psnr_{s}"] = v["psnr_by_sigma"][s]
            log.append(row)
            new_file = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            if v["psnr_mean"] > best["psnr_mean"]:
                best = {"psnr_mean": v["psnr_mean"], "step": step, "grad_err_mean": v["grad_err_mean"],
                        "psnr_by_sigma": v["psnr_by_sigma"]}
                torch.save({"model": model.state_dict(), "arch": arch, "arch_cfg": arch_cfg or {},
                            "step": step, "val": best}, best_path)
            save_ckpt(last, model, opt, step, gens, best, log)
            if max_hours and (time.time() - t0) > max_hours * 3600 and step < steps:
                status = "paused"           # session budget guard: rerun the same command to resume
                break
    info = env_info(use_gpu=device.startswith("cuda"))
    meta = {"model_id": out_dir.name, "arch": arch, "arch_cfg": arch_cfg or {}, "loss_variant": variant,
            "lambda": lam if variant == "G" else None, "train_seed": train_seed, "status": status,
            "step_reached": step, "steps_planned": steps, "batch_size": bs, "lr": cfg["lr"],
            "lr_milestones": list(cfg["lr_milestones"]), "lr_gamma": cfg["lr_gamma"],
            "params": n_params, "patch_size": P.PATCH_SIZE, "train_noise_sigma_255": list(P.TRAIN_SIGMA_RANGE_255),
            "blind": True, "input_normalization": "float32 [0,1]; noisy-image rule identical to test",
            "pad_multiple": pad_multiple(arch), "best": best, "data_stream_fingerprint": fp,
            "seeds": {k: derive_seed(train_seed, k) for k in TAGS}, "manifest_images_sha256": manifest.get("images_sha256"),
            "checkpoint_sha256": sha256_file(best_path) if best_path.exists() else None,
            "checkpoint_rule": "best mean quantized PSNR over sigma 15/25/50 on the 20 validation images",
            "source": {"dncnn": "Zhang et al. 2017, DnCNN (arXiv:1608.03167), compact variant trained by us",
                       "unet": "Ronneberger et al. 2015 (arXiv:1505.04597), architectural adaptation"}[arch],
            "environment": info, "precision": "float32 (no AMP)"}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return {"status": status, "step": step, "best": best, "params": n_params}
