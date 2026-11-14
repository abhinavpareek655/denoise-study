"""Software / hardware capture. Used for logs, meta.json and the CSV `hardware` column."""
from __future__ import annotations

import platform
import subprocess


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def env_info(use_gpu: bool = False) -> dict:
    import numpy
    import scipy
    import skimage

    info = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-image": skimage.__version__, "cpu": _cpu_model(), "git_commit": git_commit(),
            "torch": None, "cuda": None, "gpu": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    info["hardware"] = hardware_string(info, use_gpu)
    return info


def hardware_string(info: dict, use_gpu: bool) -> str:
    gpu = info.get("gpu") if use_gpu else None
    return f"cpu:{info['cpu']}|gpu:{gpu or 'none'}"
