"""Single source of truth for FROZEN protocol constants. Import these; never retype them.

Changing anything here needs both contributors' agreement (see PROJECT_CONTEXT.md, Section 22).
"""
from __future__ import annotations

# ---- Data / splits --------------------------------------------------------------------------
TEST_DATASET = "BSD68"
TRAIN_DATASET = "BSD400"
DATASET_CODES = {"BSD68": 68, "BSD400": 400}   # used in noise seeding
N_CALIBRATION = 8
N_TEST = 60
N_TRAIN = 380
N_VAL = 20
SPLITS = ("calibration", "test", "train", "validation")

# ---- Noise ----------------------------------------------------------------------------------
SIGMAS_255: tuple[int, ...] = (15, 25, 50)               # test noise on the 0..255 scale
TRAIN_SIGMA_RANGE_255: tuple[float, float] = (15.0, 50.0)  # uniform per patch; DL models are blind

# ---- Deep learning --------------------------------------------------------------------------
PATCH_SIZE = 64
TRAIN_SEEDS: tuple[int, ...] = (0, 1)
LOSS_VARIANTS: tuple[str, ...] = ("P", "G")
LAMBDA_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)
LAMBDA_FROZEN: float | None = 0.5    # A's selected shared value; confirm with A before final reporting

# ---- Metrics --------------------------------------------------------------------------------
SSIM_WIN_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_DATA_RANGE = 1.0
PSNR_CAP_DB = 100.0                  # returned (and flagged) when MSE == 0
MIN_MASK_PIXELS = 500                # smaller regions -> NaN + warning

# ---- Regions (computed on the CLEAN image) --------------------------------------------------
GRAD_SIGMA_PX = 1.0
LOCAL_STD_SIZE = 7
PCT_SMOOTH = 40
PCT_EDGE = 90
PCT_TEXTURE = 60
SENSITIVITY_DELTA = 10
# Edge percentile moved up by 10 would be 100 (degenerate) -> documented deviation: 99.
SENSITIVITY_PCTS = {
    "default": {"smooth": 40, "edge": 90, "texture": 60},
    "low": {"smooth": 30, "edge": 80, "texture": 50},
    "high": {"smooth": 50, "edge": 99, "texture": 70},
}

# ---- Reproducibility / statistics -----------------------------------------------------------
GLOBAL_SEED = 20260                  # [PROPOSED] freeze on Day 2
BOOTSTRAP_N = 10_000
BOOTSTRAP_ALPHA = 0.05

NOISY_PSNR_THEORY_DB = {15: 24.6, 25: 20.2, 50: 14.2}

# ---- Methods --------------------------------------------------------------------------------
CLASSICAL_METHODS = ("gaussian", "bilateral", "tv", "nlm")
DL_METHODS = ("dncnn", "unet")
ALL_METHODS = ("noisy",) + CLASSICAL_METHODS + DL_METHODS

# ---- Results CSV ----------------------------------------------------------------------------
CSV_COLUMNS = [
    "image_id", "split", "seed", "sigma_255", "method", "parameter_config", "model_id_optional",
    "loss_variant", "train_seed", "psnr", "ssim", "smooth_psnr", "edge_psnr", "gradient_error",
    "gradient_error_sobel", "edge_f1_optional", "runtime_seconds", "hardware",
]
CSV_EXTRA_COLUMNS = [   # appended AFTER the required columns, never inserted before
    "texture_psnr", "smooth_mae", "edge_mae", "gradient_error_edge", "gradient_error_sobel_edge",
    "n_smooth_px", "n_edge_px", "n_texture_px", "warnings",
]
PRIMARY_KEY = ["image_id", "sigma_255", "method", "parameter_config", "loss_variant", "train_seed"]

FAIRNESS_STATEMENT = (
    "Classical methods are tuned per sigma on the calibration images and are told sigma; "
    "the deep learning models are blind and were trained on sigma in [15, 50] "
    "(test sigma 15 and 50 lie at the ends of the training range)."
)
