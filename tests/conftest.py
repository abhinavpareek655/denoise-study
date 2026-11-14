import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


@pytest.fixture
def synth():
    """Deterministic synthetic image with a step edge, a ramp and a flat area (float32, [0,1])."""
    h, w = 96, 128
    img = np.full((h, w), 0.3, dtype=np.float32)
    img[:, w // 2:] = 0.8                                   # vertical step edge
    img[60:, :40] = np.linspace(0.2, 0.9, 40, dtype=np.float32)[None, :]   # ramp
    return np.clip(np.rint(img * 255) / 255, 0, 1).astype(np.float32)


def has_data() -> bool:
    from common import paths
    return (paths.data_root() / "BSD400").exists() and (paths.data_root() / "BSD68").exists()


requires_data = pytest.mark.skipif(not has_data(), reason="BSD68/BSD400 not found (set DENOISE_DATA)")
