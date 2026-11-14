"""Build artifacts/manifest.json from the raw datasets (run once by Person A, then freeze).

    python -m scripts.make_manifest [--data DIR] [--out PATH] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import paths
from common.manifest import build_manifest, save_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = a.out or paths.artifact_path("manifest.json")
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists and the manifest is FROZEN. Use --force only by agreement.")
    m = build_manifest(a.data or paths.data_root())
    save_manifest(m, out)
    print(f"wrote {out}: {m['counts']}  digest={m['images_sha256'][:12]}")


if __name__ == "__main__":
    main()
