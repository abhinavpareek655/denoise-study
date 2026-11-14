import json
from pathlib import Path

import pytest

from common.manifest import images_digest, select, validate_manifest
from common import protocol as P

MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "manifest.json"


@pytest.fixture
def manifest():
    return json.load(open(MANIFEST))


def test_counts_and_disjoint(manifest):
    validate_manifest(manifest)
    assert len(select(manifest, "BSD68", "calibration")) == P.N_CALIBRATION
    assert len(select(manifest, "BSD68", "test")) == P.N_TEST
    assert len(select(manifest, "BSD400", "train")) == P.N_TRAIN
    assert len(select(manifest, "BSD400", "validation")) == P.N_VAL


def test_digest_detects_tampering(manifest):
    manifest["images"][0]["split"] = "calibration" if manifest["images"][0]["split"] == "test" else "test"
    assert images_digest(manifest["images"]) != manifest["images_sha256"]


def test_index_is_sorted_position(manifest):
    for ds in ("BSD68", "BSD400"):
        recs = select(manifest, ds)
        assert [r["image_id"] for r in recs] == sorted(r["image_id"] for r in recs)
        assert [r["index"] for r in recs] == list(range(len(recs)))
