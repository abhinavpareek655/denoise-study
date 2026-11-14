"""Shared blind DL inference wrapper used by methods/dncnn.py and methods/unet.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common.dl_infer import restore
from common.io import check_image
from models import build_model, pad_multiple

_CACHE: dict = {}   # read-only model cache keyed by (checkpoint, device)


def load_model(model_config: dict):
    ckpt = str(model_config["checkpoint"])
    dev = str(model_config.get("device", "cpu"))
    key = (ckpt, dev)
    if key not in _CACHE:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(model_config["arch"], state.get("arch_cfg", {}))
        model.load_state_dict(state["model"])
        _CACHE[key] = model.to(dev).eval()
    return _CACHE[key]


def denoise_dl(image: np.ndarray, model_config: dict) -> np.ndarray:
    """Blind: model_config must NOT contain sigma. Keys: arch, checkpoint, device."""
    assert "sigma" not in model_config and "sigma_255" not in model_config, "DL wrappers are blind"
    check_image(image)
    model = load_model(model_config)
    dev = str(model_config.get("device", "cpu"))
    x = torch.from_numpy(image.copy())[None, None].to(dev)
    with torch.no_grad():
        out = restore(model, x, pad_multiple(model_config["arch"]), clip=True)
    return np.clip(out.squeeze().float().cpu().numpy(), 0.0, 1.0).astype(np.float32)
