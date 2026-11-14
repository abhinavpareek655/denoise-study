"""Path resolution for local runs and Kaggle. No other module may hard-code '/kaggle/...'.

Layout expected on Kaggle (the repo is cloned into /kaggle/working/denoise-study):
  * code        : <repo root>                      (this clone)
  * data        : a Kaggle dataset containing BSD68/ and BSD400/ folders (read-only)
  * artifacts   : <repo root>/artifacts            (writable; saved as notebook output)
  * shared      : optional read-only Kaggle datasets with a published `artifacts/` tree from
                  the other contributor or an earlier run (pinned dataset VERSION).

Environment variables override discovery:
  DENOISE_DATA    folder that contains BSD68/ and BSD400/
  DENOISE_SHARED  os.pathsep-separated list of extra read-only artifact roots
  DENOISE_ARTIFACTS  writable artifact root (default: <repo>/artifacts)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle").exists()


def data_root() -> Path:
    env = os.environ.get("DENOISE_DATA")
    if env:
        return Path(env)
    if ON_KAGGLE:
        hits = sorted(glob.glob("/kaggle/input/*/BSD68")) + sorted(glob.glob("/kaggle/input/*/*/BSD68"))
        if hits:
            return Path(hits[0]).parent
    return REPO_ROOT / "data"


def artifacts_root() -> Path:
    return Path(os.environ.get("DENOISE_ARTIFACTS", REPO_ROOT / "artifacts"))


def shared_roots() -> list[Path]:
    env = os.environ.get("DENOISE_SHARED", "")
    return [Path(p) for p in env.split(os.pathsep) if p]


def artifact_path(rel: str | Path, must_exist: bool = False) -> Path:
    """Writable location for `rel`. If `must_exist`, search writable root then shared roots."""
    w = artifacts_root() / rel
    if not must_exist or w.exists():
        return w
    for root in shared_roots():
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"artifact not found: {rel} (searched {artifacts_root()} and {shared_roots()})")


def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
