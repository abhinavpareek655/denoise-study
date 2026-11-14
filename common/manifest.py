"""Dataset manifest: frozen record of which image is used for what (built once by Person A)."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import protocol as P
from common.io import sha256_file


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def images_digest(images: list[dict]) -> str:
    """Hash of the image records only (not timestamps); stored in results to prove same manifest."""
    blob = json.dumps(images, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_manifest(data_root: Path, global_seed: int = P.GLOBAL_SEED) -> dict:
    from PIL import Image

    records: list[dict] = []
    for dataset in (P.TEST_DATASET, P.TRAIN_DATASET):
        files = _list_images(Path(data_root) / dataset)
        n = len(files)
        if dataset == P.TEST_DATASET:
            sel_rng = np.random.default_rng(np.random.SeedSequence([global_seed, 1]))
            chosen = set(sel_rng.choice(n, size=P.N_CALIBRATION, replace=False).tolist())
            label_sel, label_rest = "calibration", "test"
        else:
            sel_rng = np.random.default_rng(np.random.SeedSequence([global_seed, 2]))
            chosen = set(sel_rng.choice(n, size=P.N_VAL, replace=False).tolist())
            label_sel, label_rest = "validation", "train"
        for idx, f in enumerate(files):
            with Image.open(f) as im:
                w, h = im.size
            records.append({
                "image_id": f.stem, "dataset": dataset,
                "split": label_sel if idx in chosen else label_rest,
                "index": idx, "relative_path": f"{dataset}/{f.name}",
                "sha256": sha256_file(f), "height": h, "width": w,
            })
    try:
        import numpy, scipy, skimage, PIL  # noqa: F401
        versions = {"python": platform.python_version(), "numpy": numpy.__version__,
                    "scipy": scipy.__version__, "scikit-image": skimage.__version__,
                    "pillow": PIL.__version__}
    except Exception:  # pragma: no cover
        versions = {"python": platform.python_version()}
    m = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "global_seed": global_seed,
        "source": {
            "BSD68": "68 grayscale test images (Berkeley Segmentation Dataset), as distributed in "
                     "https://github.com/cszn/DnCNN (testsets/BSD68).",
            "BSD400": "400 grayscale 180x180 training images (Berkeley Segmentation Dataset), as "
                      "distributed in https://github.com/cszn/DnCNN "
                      "(TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400). Academic use; "
                      "cite Martin et al. 2001 (BSDS).",
        },
        "preprocessing": "Grayscale (PIL convert('L')), uint8 -> float32 /255 in [0,1]. No resizing "
                         "or cropping. Originals untouched.",
        "selection_rule": {
            "calibration": f"8 of the 68 BSD68 images, drawn without replacement with "
                           f"default_rng(SeedSequence([{global_seed}, 1])); remaining 60 are test.",
            "validation": f"20 of the 400 BSD400 images, drawn without replacement with "
                          f"default_rng(SeedSequence([{global_seed}, 2])); remaining 380 are train.",
            "index": "0-based position of the file in the lexicographically sorted list within a dataset.",
            "noise_seed": "SeedSequence([global_seed, dataset_code(BSD68=68, BSD400=400), index, sigma_255])",
        },
        "software_versions": versions,
        "counts": {},
        "images": records,
    }
    m["counts"] = count_splits(m)
    m["images_sha256"] = images_digest(records)
    validate_manifest(m)
    return m


def count_splits(m: dict) -> dict:
    out: dict[str, int] = {}
    for r in m["images"]:
        out[r["split"]] = out.get(r["split"], 0) + 1
    return out


def validate_manifest(m: dict, data_root: Path | None = None) -> None:
    c = count_splits(m)
    want = {"calibration": P.N_CALIBRATION, "test": P.N_TEST, "train": P.N_TRAIN, "validation": P.N_VAL}
    assert c == want, f"split counts {c} != {want}"
    keys = [(r["dataset"], r["image_id"]) for r in m["images"]]
    assert len(keys) == len(set(keys)), "duplicate (dataset, image_id)"
    for r in m["images"]:
        assert r["split"] in P.SPLITS
        if r["dataset"] == P.TEST_DATASET:
            assert r["split"] in ("calibration", "test"), "BSD68 may only be calibration/test"
        else:
            assert r["split"] in ("train", "validation"), "BSD400 may only be train/validation"
    hashes_68 = {r["sha256"] for r in m["images"] if r["dataset"] == P.TEST_DATASET}
    hashes_400 = {r["sha256"] for r in m["images"] if r["dataset"] == P.TRAIN_DATASET}
    assert not hashes_68 & hashes_400, "a BSD68 file is byte-identical to a BSD400 file"
    if "images_sha256" in m:
        assert m["images_sha256"] == images_digest(m["images"]), "manifest digest mismatch"
    if data_root is not None:
        for r in m["images"]:
            assert sha256_file(Path(data_root) / r["relative_path"]) == r["sha256"], \
                f"hash mismatch for {r['relative_path']}"


def save_manifest(m: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
        f.write("\n")


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def select(m: dict, dataset: str | None = None, split: str | None = None) -> list[dict]:
    return [r for r in m["images"]
            if (dataset is None or r["dataset"] == dataset) and (split is None or r["split"] == split)]
