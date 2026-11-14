import numpy as np
import pandas as pd

from common import protocol as P
from common.evaluator import integrity_check, order_columns, score_image
from common.regions import fit_thresholds, make_masks
from common.stats import bootstrap_ci, paired_diff, seed_spread
from analysis.equal_smoothing import bootstrap_matched, interp_at, rising_branch


def test_csv_column_order_exact():
    df = order_columns(pd.DataFrame({"image_id": ["a"], "psnr": [1.0]}))
    assert list(df.columns) == P.CSV_COLUMNS + P.CSV_EXTRA_COLUMNS
    assert P.CSV_COLUMNS[0] == "image_id" and P.CSV_COLUMNS[-1] == "hardware"


def test_score_image_fields(synth):
    th = fit_thresholds([synth], P.SENSITIVITY_PCTS["default"])
    f, w = score_image(synth, synth, make_masks(synth, th))
    assert f["psnr"] == P.PSNR_CAP_DB and "psnr_capped" in w
    assert f["gradient_error"] == 0.0


def _row(**kw):
    base = dict(image_id="a", split="test", sigma_255=15, method="gaussian", parameter_config="p",
                loss_variant="none", train_seed=np.nan, psnr=30.0, ssim=0.8, gradient_error=0.01,
                gradient_error_sobel=0.01)
    base.update(kw)
    return base


def test_integrity_detects_duplicates_and_nonfinite():
    df = pd.DataFrame([_row(), _row()])
    assert any("duplicate" in p for p in integrity_check(df))
    assert any("non-finite" in p for p in integrity_check(pd.DataFrame([_row(psnr=np.inf)])))
    assert integrity_check(pd.DataFrame([_row()])) == []
    assert any("non-test" in p for p in integrity_check(pd.DataFrame([_row(split="calibration")])))


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    d = np.random.default_rng(0).normal(0.5, 1.0, 60)
    m, lo, hi = bootstrap_ci(d, n_boot=2000)
    assert lo < m < hi and (m, lo, hi) == bootstrap_ci(d, n_boot=2000)


def test_paired_diff_and_seed_spread():
    rows = []
    for i in range(5):
        for meth, lv, ts, v in [("unet", "P", 0, 30.0), ("unet", "P", 1, 30.2), ("unet", "G", 0, 30.5),
                                ("unet", "G", 1, 30.7), ("gaussian", "none", np.nan, 28.0)]:
            rows.append(_row(image_id=f"i{i}", method=meth, loss_variant=lv, train_seed=ts, psnr=v + i * 0.1))
    df = pd.DataFrame(rows)
    d = paired_diff(df, "unet_G", "unet_P", "psnr", 15)
    assert np.allclose(d.values, 0.5) and len(d) == 5
    sp = seed_spread(df, "unet", "P", "psnr", 15)
    assert abs(sp["max"] - sp["min"] - 0.2) < 1e-9


def test_interp_no_extrapolation():
    x, y = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    assert interp_at(x, y, 2.5) == 25.0
    assert np.isnan(interp_at(x, y, 0.5)) and np.isnan(interp_at(x, y, 3.5))
    assert interp_at([3.0, 1.0, 2.0], [30.0, 10.0, 20.0], 1.5) == 15.0      # unsorted input ok


def test_bootstrap_matched_ci():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(30):
        off = rng.normal(0, 0.2)
        for v in (1.0, 2.0, 3.0):
            rows.append({"image_id": f"i{i}", "setting_value": v, "smooth_psnr": 30 + 2 * v + off,
                         "edge_psnr": 25 - v + off})
    pt, lo, hi = bootstrap_matched(pd.DataFrame(rows), 34.0, n_boot=500)
    assert lo <= pt <= hi and abs(pt - 23.0) < 0.3
    assert np.isnan(bootstrap_matched(pd.DataFrame(rows), 99.0, n_boot=50)[0])


def test_rising_branch_cuts_after_peak():
    rows = []
    for i in range(10):
        for v, x in [(1.0, 28.0), (2.0, 31.0), (3.0, 33.0), (4.0, 32.0), (5.0, 30.0)]:   # folds back after v=3
            rows.append({"image_id": f"i{i}", "setting_value": v, "smooth_psnr": x, "edge_psnr": 30 - v})
    df = pd.DataFrame(rows)
    assert rising_branch(df) == [1.0, 2.0, 3.0]
    pt, _, _ = bootstrap_matched(df, 31.0, n_boot=50)
    assert pt == 28.0                         # reads v=2 on the rising branch, not the falling v=4.5
    assert np.isnan(bootstrap_matched(df, 29.0 - 5.0, n_boot=20)[0])
