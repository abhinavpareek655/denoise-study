"""Method registry and helpers shared by every script."""
from __future__ import annotations

import importlib

from common.protocol import ALL_METHODS


def config_id(config: dict) -> str:
    """Stable short string for the `parameter_config` column (sorted keys, floats via repr)."""
    parts = []
    for k in sorted(config):
        v = config[k]
        parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return ";".join(parts)


def get_module(method: str):
    if method not in ALL_METHODS or method == "noisy":
        raise KeyError(f"unknown method {method!r}; choose from {ALL_METHODS[1:]}")
    return importlib.import_module(f"methods.{method}")
