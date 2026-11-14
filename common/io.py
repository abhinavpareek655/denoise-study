"""Image I/O and shared validation helpers. Interface scale: float32, (H, W), [0, 1]."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def check_image(x: np.ndarray, name: str = "image") -> None:
    assert isinstance(x, np.ndarray), f"{name}: not an ndarray"
    assert x.ndim == 2, f"{name}: expected (H, W), got {x.shape}"
    assert x.dtype == np.float32, f"{name}: dtype {x.dtype}, expected float32"
    assert np.isfinite(x).all(), f"{name}: non-finite values"
    assert x.min() >= 0.0 and x.max() <= 1.0, f"{name}: values outside [0, 1]"


def quantize8(x01: np.ndarray) -> np.ndarray:
    """Clip, scale and round to uint8 (the 'output rule' used before every metric)."""
    return np.rint(np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_gray01(path: str | Path) -> np.ndarray:
    """Grayscale float32 in [0,1]. No resizing or cropping."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr.astype(np.float32) / 255.0).astype(np.float32)


def save_png8(path: str | Path, img01: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantize8(img01), mode="L").save(path, format="PNG")


def load_png8(path: str | Path) -> np.ndarray:
    return load_gray01(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
