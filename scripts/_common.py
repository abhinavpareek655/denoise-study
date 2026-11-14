"""Small helpers shared by the CLI scripts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from common import paths
from common.manifest import load_manifest


def load_yaml(p: str | Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def get_manifest() -> dict:
    return load_manifest(paths.artifact_path("manifest.json", must_exist=True))


def clean_path(rec: dict) -> Path:
    return paths.data_root() / rec["relative_path"]


def noisy_path(rec: dict, sigma: int) -> Path:
    return paths.artifact_path(f"noisy/sigma{sigma}/{rec['image_id']}.png", must_exist=True)


def read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
