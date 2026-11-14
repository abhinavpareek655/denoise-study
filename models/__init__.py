from __future__ import annotations

import importlib

import torch.nn as nn


def build_model(arch: str, cfg: dict | None = None) -> nn.Module:
    return importlib.import_module(f"models.{arch}").build(cfg or {})


def pad_multiple(arch: str) -> int:
    return importlib.import_module(f"models.{arch}").PAD_MULTIPLE


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
