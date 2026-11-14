# PROJECT_CONTEXT.md
## Shared engineering contract for Person A and Person B (and any LLM that writes code for them)

**Project:** An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness, Detail Preservation, and Computational Efficiency.

**Purpose of this file.** It is the single source of truth. Paste it (whole) at the start of every LLM coding session, and commit it to the repo root. If code, a chat answer, or a person disagrees with this file, this file wins until both people edit it together. Changing a **FROZEN** item requires both people to agree and a dated line in the changelog (Section 22).

**Provenance.** Built from the Project Plan and the Person B guide. Person A's own guide was not available when this was written, so everything about A's work is taken from the plan and from B's guide, and every convention I had to invent is marked **[PROPOSED]**. Items marked **[PROPOSED]** must be confirmed on Day 2 and then flipped to FROZEN.

**Repo as built.** Section 25 is the authoritative file map of the actual repository (it supersedes the
file paths in Sections 3 and 4 wherever they differ) and records decisions made while building it.

---

## 0. Instructions to the LLM that generates code from this file

Follow these rules in every code-generation task for this project.

1. **Never change a FROZEN value.** Constants live in `common/protocol.py`; import them, never retype them. If a task seems to require changing one, stop and say so.
2. **Respect ownership (Section 3).** Only edit the files your person owns. Shared code lives in `common/`; changes there need both people's sign-off. Never import from the other person's private modules; talk only through `common/` and files in `artifacts/`.
3. **Use the exact interfaces (Section 6)**: image in `float32` `[0,1]`, shape `(H, W)`, out the same. No in-place changes, finite values, explicit clip.
4. **Never use test images for tuning, thresholds, checkpoint choice, or lambda choice.** Calibration images for classical tuning and mask thresholds; the 20 validation images for checkpoints and the lambda pilot; BSD68 test images only for final evaluation.
5. **Never pass sigma into a deep learning wrapper or model.** Classical wrappers may receive sigma-dependent parameters through their config, nothing else.
6. **Deterministic by default.** Every random draw comes from a named, seeded generator (Section 7). No `np.random.seed`, no unseeded `random`, no `hash()` for seeding.
7. **Write the fair-comparison statement** (Section 13) next to any table or plot that compares classical and learned methods.
8. **Prefer the reference snippets in this file** (noise, metrics, masks, Sobel loss, wrapper). Re-derive nothing that is already specified. If you must deviate, state the reason in a comment.
9. **Add a test** for every new function (Section 17). Code without a test is not done.
10. **State assumptions.** If a detail is not in this file (marked OPEN in Section 21), do not silently invent it: pick the stated default, put it in the config file, and mention it in your reply.
11. **Style:** Section 16.
12. Do not add optional extensions (Section 20) unless the core deliverables are complete and the user asks.
13. **All heavy compute runs on Kaggle** (Section 24): use `common/paths.py` for every path, make training resumable, never hard-code `/kaggle/...`, and keep notebooks thin.

---

## 1. Research question and framing

How do classical and modern denoisers trade off noise suppression, edge/texture retention, and runtime under controlled AWGN, and how does that trade-off depend on noise level and image content?

- The contribution is a **controlled, reproducible investigation**, not a leaderboard.
- AWGN on grayscale images is a controlled proxy, not a model of real camera noise (which can be signal-dependent, processed, compressed, spatially varying).
- Two networks cannot represent all modern denoisers; six methods do not represent all of denoising. Do not claim exact reproduction of, or comparability to, the inspiration paper.
- Because DnCNN and the U-Net differ in depth, skip connections and normalization all at once, **never attribute differences to "architecture type" alone**. Report parameter counts.
- The original U-Net paper (https://arxiv.org/abs/1505.04597) is about biomedical segmentation. Our U-Net is an architectural adaptation for denoising; do not claim the paper evaluated denoising.
- Expected results are hypotheses, not required outcomes. Results may contradict them.

### Methods (six, all core)

| # | Method | `method` string in CSV | Owner | Implementation |
|---|---|---|---|---|
| 1 | Gaussian smoothing | `gaussian` | A | Reuse existing from-scratch code behind the interface |
| 2 | Bilateral filter | `bilateral` | B | Reuse existing from-scratch code behind the interface |
| 3 | Total variation | `tv` | B | Library OK if labelled (e.g. scikit-image `denoise_tv_chambolle`) |
| 4 | Non-Local Means | `nlm` | A | Library OK if labelled |
| 5 | DnCNN (residual CNN) | `dncnn` | A | Written by us in PyTorch, trained on Kaggle GPU |
| 6 | Compact residual U-Net | `unet` | B | Written by us in PyTorch, trained on Kaggle GPU |

Extra pseudo-method for sanity and baselines: `noisy` (identity; the noisy input scored against clean).

---

## 2. FROZEN protocol (identical in all documents)

| Item | Decision |
|---|---|
| Test data | BSD68, grayscale. Image IDs frozen in the manifest: **8 calibration images** (classical tuning and mask thresholds) and **60 test images**. Test images are never used for tuning, checkpoint choice, or threshold choice. |
| DL training data | **BSD400** (400 images). Split **before** patch extraction: **380 train, 20 validation** (checkpoint choice and lambda pilot). BSD68 is never used for training or model selection. |
| Preprocessing | Grayscale, `float32` in [0,1]. Interface scale is [0,1] everywhere. Classical parameters that live on the 0 to 255 scale (bilateral range sigma) are converted inside the wrapper and stored in documented units. |
| Test noise | AWGN, sigma = **15, 25, 50** on the 0 to 255 scale (std = sigma/255 on [0,1]). **One deterministic noisy realization per image and sigma**, reused by every method. |
| Training noise | Online AWGN, sigma sampled **uniformly from [15, 50] per patch**. DL models are **blind** (not told sigma). Test sigma 15 and 50 sit at the ends of the range; report as a limitation. |
| Noisy-image rule | `noisy = clip(round(255*clean + N(0, sigma^2)), 0, 255) / 255`, applied **identically in training and test**. Check pre-clipping mean and std. |
| Output rule | Every method outputs float in [0,1] with explicit clipping. For metrics, all outputs are rounded to 8-bit first. |
| DL recipe | 64 x 64 patches, flips and 90-degree rotations, Adam, L1 pixel loss, compact networks. Same patches or steps, validation images, checkpoint rule and noise sampler for both networks. Parameter counts reported. **2 training seeds** per configuration. |
| Core ablation | **P** = pixel loss only. **G** = pixel loss + lambda x Sobel gradient loss (L1 on Sobel x and y responses, kernels divided by 8). Lambda chosen by a short pilot on the validation images only (grid **0.1, 0.5, 1.0**), then fixed. Each person runs P and G on their own network, 2 seeds each. |
| Fairness rule | Classical methods are tuned **per sigma** on the calibration images (they are told sigma). DL models are blind. State this next to every comparison. A sigma-specific DL model is optional. |
| Metrics | PSNR (fixed range, quantized output), SSIM (Gaussian window 11, sigma 1.5, data range 1), runtime, region metrics, gradient error, optional Canny agreement. |
| Region masks | Built from the **clean** image with a **Gaussian-derivative** gradient (sigma 1 px) and local standard deviation (7 x 7). Defaults, fixed on the calibration images by percentile: smooth = gradient and local std both at or below their 40th percentile; edge = gradient at or above the 90th percentile; optional texture = local std at or above the 60th percentile and not edge. Report pixel counts; test sensitivity by moving percentiles by 10 points. |
| Gradient error | Headline: Gaussian-derivative gradient (sigma 1 px), mean absolute difference of gradient magnitude, globally and inside edge regions. A Sobel-based version only as a clearly labelled **"same operator as the G loss"** number (G is trained on Sobel, so it is favoured by construction). |
| Statistics | Images are the experimental unit. Paired differences per image, bootstrap 95% CIs over images, seed spread for DL models. |

---

## 3. Ownership map

| Area | Owner | Files (under repo root) |
|---|---|---|
| Manifest, splits, noise generator, noisy test sets, I/O tests | **A** | `common/manifest.py`, `common/noise.py`, `common/io.py`, `scripts/make_manifest.py`, `scripts/make_noisy.py` |
| Gaussian, NLM wrappers and tuning | **A** | `methods/gaussian.py`, `methods/nlm.py` |
| DnCNN model, training, wrapper, P and G, 2 seeds each | **A** | `models/dncnn.py`, `methods/dncnn.py`, `configs/dncnn_*.yaml` |
| Runtime benchmarking harness (all methods) | **A** | `analysis/runtime.py` |
| Statistics: paired differences, bootstrap CIs, seed spread, P-vs-G tests | **A** | `common/stats.py`, `analysis/stats_report.py` |
| Plots 2 to 4 (PSNR and SSIM vs sigma, quality vs runtime), parameter-response curves | **A** | `plots/plot2_*.py` to `plot4_*.py`, `plots/param_response.py` |
| Theory notes A (linear-filter variance, NLM weights, residual learning, receptive field) | **A** | `docs/theory_A.md` |
| Bilateral and TV wrappers and tuning | **B** | `methods/bilateral.py`, `methods/tv.py` |
| U-Net model, training, wrapper, P and G, 2 seeds each | **B** | `models/unet.py`, `methods/unet.py`, `configs/unet_*.yaml` |
| Evaluator core: PSNR, SSIM, CSV, integrity checks | **B** | `common/metrics.py`, `evaluate/evaluate.py`, `evaluate/integrity.py` |
| Region masks, regional metrics, gradient error | **B** | `common/regions.py`, `scripts/make_masks.py` |
| Equal-smoothing analysis | **B** | `analysis/equal_smoothing.py` |
| Failure gallery, plots 1, 5, 7 | **B** | `plots/plot1_*.py`, `plot5_*.py`, `plot7_*.py`, `analysis/failure_gallery.py` |
| Theory notes B (bilateral weights, TV objective and solver, skip connections, Sobel loss circularity) | **B** | `docs/theory_B.md` |
| **Shared, both sign off** | A+B | `common/protocol.py`, `common/registry.py`, `common/dl_train.py`, `common/losses.py`, `common/plotting.py`, `common/versions.py`, this file |

Both people: understand every method, freeze the protocol together, cross-check outputs, co-write conclusions and slides.

**Plot numbering:** 1 clean/noisy/all-method grid (B); 2 to 4 PSNR/SSIM vs sigma and quality vs runtime (A); 5 region-wise results (B); 6 parameter-response curves (A); 7 failure gallery (B). **[PROPOSED]** for plot 6 = A's parameter-response curves; adjust if A's guide numbers them differently.

---

## 4. Repository layout

```
repo/
  PROJECT_CONTEXT.md            # this file
  README.md                     # reproduction steps (one-command reproduction)
  requirements.txt              # pinned versions
  pyproject.toml                # ruff / pytest config
  Makefile                      # or scripts/run_all.py with --stage
  common/
    protocol.py                 # ALL frozen constants (Section 5)
    io.py                       # load_gray01, save_png8, load_png8
    noise.py                    # add_awgn_8bit, torch sampler
    manifest.py                 # read/write/validate manifest
    metrics.py                  # quantize8, psnr, ssim, mae
    regions.py                  # grad_mag, local_std, masks, regional metrics, gradient error
    losses.py                   # pixel + Sobel gradient loss
    dl_train.py                 # shared trainer used by BOTH networks
    registry.py                 # METHODS dict, method ids
    stats.py                    # bootstrap (A)
    plotting.py                 # colors, markers, style
    versions.py                 # software / hardware capture
  methods/                      # one wrapper per method, interface in Section 6
  models/                       # torch nn.Module definitions only
  configs/                      # YAML: grids, frozen settings, training configs
  evaluate/                     # evaluator, integrity checks
  analysis/                     # equal smoothing, stats report, runtime, gallery
  plots/
  scripts/                      # CLI entry points
  tests/                        # pytest
  docs/                         # theory notes, adaptation card, model provenance
  data/                         # raw datasets (gitignored): BSD400/, BSD68/
  artifacts/                    # generated, versioned by hash
    manifest.json
    noisy/sigma{15,25,50}/{image_id}.png
    outputs/{method_id}/sigma{15,25,50}/{image_id}.png   # quantized 8-bit outputs
    masks/thresholds.json
    masks/{image_id}.npz
    tuning/{method}_sigma{S}_grid.csv
    tuning/{method}_frozen.json
    checkpoints/{model_id}/best.pt, last.pt, config.yaml, train_log.csv, meta.json
    results/results.csv         # main per-image CSV (Section 9)
    results/equal_smoothing.csv
    results/runtime.csv
    logs/
    figures/
```

`data/` is never committed. `artifacts/` holds generated files; each has a producing script.

---

## 5. Constants: `common/protocol.py` (copy exactly)

```python
"""Single source of truth for frozen protocol constants. Import, never retype."""
from __future__ import annotations

SIGMAS_255: tuple[int, ...] = (15, 25, 50)          # test noise, 0..255 scale
TRAIN_SIGMA_RANGE_255: tuple[float, float] = (15.0, 50.0)  # uniform per patch, blind DL
N_CALIBRATION = 8
N_TEST = 60
N_TRAIN = 380
N_VAL = 20
PATCH_SIZE = 64
TRAIN_SEEDS: tuple[int, ...] = (0, 1)              # 2 seeds per configuration
LOSS_VARIANTS: tuple[str, ...] = ("P", "G")
LAMBDA_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)   # pilot on validation only
LAMBDA_FROZEN: float | None = None                 # set after pilot (both networks use the SAME value)

# Metrics
SSIM_WIN_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_DATA_RANGE = 1.0
PSNR_CAP_DB = 100.0                                # returned when MSE == 0, and logged

# Regions (computed on the CLEAN image)
GRAD_SIGMA_PX = 1.0                                # Gaussian-derivative scale
LOCAL_STD_SIZE = 7                                 # 7 x 7 uniform window
PCT_SMOOTH = 40                                    # grad <= p40 AND std <= p40
PCT_EDGE = 90                                      # grad >= p90
PCT_TEXTURE = 60                                   # std >= p60 AND NOT edge
SENSITIVITY_DELTA = 10                             # move percentiles by +-10 points

# Reproducibility  [PROPOSED value; freeze Day 2]
GLOBAL_SEED = 20260

# Sanity yardsticks: noisy-input PSNR (dB), slightly higher in practice due to clipping
NOISY_PSNR_THEORY_DB = {15: 24.6, 25: 20.2, 50: 14.2}

# Statistics
BOOTSTRAP_N = 10_000                               # [PROPOSED]
BOOTSTRAP_ALPHA = 0.05                             # 95% CI

FAIRNESS_STATEMENT = (
    "Classical methods are tuned per sigma on the calibration images and are told sigma; "
    "the deep learning models are blind and were trained on sigma in [15, 50] "
    "(test sigma 15 and 50 lie at the ends of the training range)."
)
```

---

## 6. Interfaces (all methods must conform)

```python
# Classical (may use sigma-dependent values only through `config`)
def denoise(image: np.ndarray, config: dict) -> np.ndarray: ...

# Learned, BLIND: must not receive or use sigma
def denoise_dl(image: np.ndarray, model_config: dict) -> np.ndarray: ...
```

Hard requirements for every wrapper:

- Input: `np.float32`, shape `(H, W)`, values in [0,1], finite. Output: **same shape, `np.float32`, finite, explicitly clipped** with `np.clip(out, 0.0, 1.0)`.
- **No in-place modification** of the input. Work on a copy.
- Wrapper documents (module docstring): normalization, padding/cropping, border handling, library name and version, parameter units, runtime inclusion rules.
- Each module in `methods/` also exposes:
  ```python
  METHOD = "bilateral"                       # matches CSV `method`
  def default_grid() -> list[dict]: ...      # declared tuning grid (also saved in configs/)
  def config_id(config: dict) -> str: ...    # short stable string stored in `parameter_config`
  ```
- Register in `common/registry.py`:
  ```python
  METHODS = {"gaussian": ..., "bilateral": ..., "tv": ..., "nlm": ..., "dncnn": ..., "unet": ...}
  ```
- Units: interface scale is [0,1]. Parameters defined on the 0 to 255 scale are **stored in config in 255-units with a `_255` suffix** and converted inside the wrapper. Example: bilateral `sigma_r_255` (stored) -> `sigma_r_255 / 255` (used on [0,1] data). Spatial scales are always pixels, suffix `_px`.
- Config keys use `snake_case`; configs are plain dicts / YAML and JSON-serialisable.

### 6.1 Reference check helper (`common/io.py`)

```python
def check_image(x: np.ndarray, name: str = "image") -> None:
    assert isinstance(x, np.ndarray), f"{name}: not ndarray"
    assert x.ndim == 2, f"{name}: expected (H, W), got {x.shape}"
    assert x.dtype == np.float32, f"{name}: dtype {x.dtype}"
    assert np.isfinite(x).all(), f"{name}: non-finite values"
    assert x.min() >= 0.0 and x.max() <= 1.0, f"{name}: out of [0,1]"
```

### 6.2 Image I/O (`common/io.py`)

- `load_gray01(path) -> float32 (H, W) in [0,1]` via PIL: `Image.open(p).convert("L")`, then `np.asarray(..., dtype=np.uint8).astype(np.float32) / 255.0`. **No resizing, no cropping** of BSD68 test images; originals untouched.
- `save_png8(path, img01)` writes `quantize8(img01)` as lossless PNG. Because the noisy-image rule yields integer 8-bit levels, saved noisy PNGs are **exact**.
- `load_png8(path)` returns float32 [0,1].

---

## 7. Seeds, noise generation and the manifest

### 7.1 Manifest (`artifacts/manifest.json`, built once by A, then frozen)

Fields: `created_utc`, `global_seed`, `source` (dataset names and attribution), `preprocessing` description, `software_versions`, and `images`: a list of records

```json
{"image_id": "test003", "dataset": "BSD68", "split": "test", "index": 3,
 "relative_path": "data/BSD68/test003.png", "sha256": "...", "height": 321, "width": 481}
```

- `image_id` = file stem. `index` = 0-based position in the **sorted** list of image IDs within the dataset, used for seeding.
- `split` in `{"calibration", "test", "train", "validation"}`. BSD68 -> 8 calibration + 60 test; BSD400 -> 380 train + 20 validation. The calibration/validation selection is a **seeded random draw recorded once** in the manifest **[PROPOSED]**; the split is done **before** patch extraction.
- Validation: counts match Section 5 constants; no ID appears in two splits; hashes match files; BSD68 IDs never appear in BSD400.

### 7.2 Test-noise generator (`common/noise.py`) , reference implementation

```python
import numpy as np

def noise_rng(dataset: str, image_index: int, sigma_255: int, global_seed: int = GLOBAL_SEED) -> np.random.Generator:
    """Deterministic per (dataset, image, sigma). Never use Python hash().
    SeedSequence([global_seed, DATASET_CODES[dataset] (BSD68=68, BSD400=400), index, sigma_255])."""
    ...

def add_awgn_8bit(clean01: np.ndarray, sigma_255: float, rng: np.random.Generator):
    """noisy = clip(round(255*clean + N(0, sigma^2)), 0, 255) / 255.
    Returns (noisy01 float32, pre_clip_noise float64) so mean/std can be checked."""
    clean255 = np.rint(clean01.astype(np.float64) * 255.0)       # clean is already on the 8-bit grid
    noise = rng.normal(0.0, sigma_255, size=clean255.shape)
    noisy255 = np.clip(np.rint(clean255 + noise), 0.0, 255.0)
    return (noisy255 / 255.0).astype(np.float32), noise

def check_noise_stats(noise: np.ndarray, sigma_255: float, k: float = 4.0) -> None:
    n = noise.size
    se_mean = sigma_255 / np.sqrt(n)
    assert abs(noise.mean()) < k * se_mean, "pre-clip noise mean off"
    assert abs(noise.std() / sigma_255 - 1.0) < 0.02, "pre-clip noise std off"
```

- Test noisy files are generated **once** by `scripts/make_noisy.py`, saved to `artifacts/noisy/sigma{S}/{image_id}.png`, and **every method reads those files**. No method generates its own test noise.
- Calibration images use the same generator and the same rule (same seeds), saved the same way.

### 7.3 Training-noise sampler (shared, in `common/dl_train.py`)

Identical rule as test noise, in torch, sigma per patch:

```python
def add_awgn_torch(clean01: torch.Tensor, gen: torch.Generator):
    """clean01: (B,1,H,W) on the 8-bit grid. sigma ~ U[15, 50] per patch (blind training).
    Returns (noisy01, sigma_255). Noise is drawn on CPU for determinism, then moved."""
    B = clean01.shape[0]
    lo, hi = TRAIN_SIGMA_RANGE_255
    sigma = lo + (hi - lo) * torch.rand(B, 1, 1, 1, generator=gen)
    noise = torch.randn(clean01.shape, generator=gen) * sigma
    clean255 = torch.round(clean01.cpu() * 255.0)
    noisy = torch.clamp(torch.round(clean255 + noise), 0.0, 255.0) / 255.0
    return noisy.to(clean01.device), sigma
```

(`sigma` is returned only for logging/diagnostics; **it is never fed to the model**.)

### 7.4 Seed discipline for DL (needed for a clean P vs G comparison)

For a given `train_seed`, P and G must be **identical in everything except the loss**: same weight initialization, same data order and patch crops/augmentations, same noise draws, same number of steps. Implementation: three independent generators seeded from `train_seed` -- `init` (weights), `data` (patch positions, augmentation, batch order), `noise` (sigma and noise draws) -- so that loss-dependent computation never consumes their random state. The same generators and trainer are used for both networks (only the model differs). Record seeds in `meta.json`.

---

## 8. Metrics (`common/metrics.py`) , reference implementation

```python
import numpy as np
from skimage.metrics import structural_similarity
from common.protocol import *

def quantize8(x01: np.ndarray) -> np.ndarray:
    """Clip, scale, round to uint8. Applied to every output AND the clean reference before any metric."""
    return np.rint(np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)

def _q(x01):  # quantized back to float64 [0,1]
    return quantize8(x01).astype(np.float64) / 255.0

def psnr(clean01, out01, mask=None) -> float:
    d = (_q(clean01) - _q(out01)) ** 2
    mse = d.mean() if mask is None else d[mask].mean()
    return PSNR_CAP_DB if mse == 0 else float(10.0 * np.log10(1.0 / mse))   # peak = 1.0; log if capped

def mae(clean01, out01, mask=None) -> float:
    d = np.abs(_q(clean01) - _q(out01))
    return float(d.mean() if mask is None else d[mask].mean())

def ssim(clean01, out01) -> float:
    return float(structural_similarity(
        _q(clean01), _q(out01), data_range=SSIM_DATA_RANGE,
        gaussian_weights=True, sigma=SSIM_SIGMA, win_size=SSIM_WIN_SIZE,
        use_sample_covariance=False))
```

Notes:
- Peak is fixed at 1.0 on the [0,1] scale after 8-bit rounding (equivalent to peak 255 on 8-bit values). Never use `data_range=` computed from the image.
- SSIM: Gaussian window 11, sigma 1.5, data range 1, population covariance (the standard Wang et al. setting). scikit-image trims a border of `(win_size-1)//2` pixels internally; document this. Validate once against a second reference implementation (unit test).
- Regional PSNR/MAE use the same functions with a boolean `mask`; **no border cropping** for regional metrics.
- If a mask has fewer than `MIN_MASK_PIXELS` (**[PROPOSED]** 500), return `NaN` and log; never silently average tiny regions.
- Always the same clean image and the same noisy realization for all methods.

---

## 9. Region masks and gradient error (`common/regions.py`)

### 9.1 Reference implementation

```python
import numpy as np
from scipy import ndimage as ndi
from common.protocol import *

def grad_mag_gauss(img01: np.ndarray, sigma_px: float = GRAD_SIGMA_PX) -> np.ndarray:
    """Gaussian-derivative gradient magnitude (headline operator)."""
    return ndi.gaussian_gradient_magnitude(img01.astype(np.float64), sigma=sigma_px, mode="reflect")

def local_std(img01: np.ndarray, size: int = LOCAL_STD_SIZE) -> np.ndarray:
    x = img01.astype(np.float64)
    m = ndi.uniform_filter(x, size=size, mode="reflect")
    m2 = ndi.uniform_filter(x * x, size=size, mode="reflect")
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))

def fit_thresholds(clean_calib: list[np.ndarray], pcts: dict) -> dict:
    """Percentiles pooled over ALL pixels of the 8 CALIBRATION images [PROPOSED pooling].
    pcts example: {'smooth': 40, 'edge': 90, 'texture': 60}. Never fit on test images."""
    g = np.concatenate([grad_mag_gauss(c).ravel() for c in clean_calib])
    s = np.concatenate([local_std(c).ravel() for c in clean_calib])
    return {"grad_smooth": np.percentile(g, pcts["smooth"]), "std_smooth": np.percentile(s, pcts["smooth"]),
            "grad_edge": np.percentile(g, pcts["edge"]), "std_texture": np.percentile(s, pcts["texture"]), "pcts": pcts}

def make_masks(clean01: np.ndarray, th: dict) -> dict:
    g, s = grad_mag_gauss(clean01), local_std(clean01)
    edge = g >= th["grad_edge"]
    smooth = (g <= th["grad_smooth"]) & (s <= th["std_smooth"])
    texture = (s >= th["std_texture"]) & ~edge
    return {"smooth": smooth, "edge": edge, "texture": texture}
```

- Masks depend only on the **clean** image, so they are the same for all sigmas and methods. Save thresholds to `artifacts/masks/thresholds.json` and per-image masks to `artifacts/masks/{image_id}.npz`.
- Report **pixel counts per region** per image and pooled. Smooth and edge are disjoint by construction in practice; check and log any overlap. Do not conflate textured regions with boundaries (texture excludes edge).
- **Sensitivity analysis:** repeat with percentiles moved by 10 points: smooth 30 and 50; edge 80 and **99** (100 would be degenerate: documented deviation **[PROPOSED]**); texture 50 and 70. Report whether method rankings change.
- Tiny regions give unstable scores (Section 8 rule).

### 9.2 Gradient error

```python
def gradient_error(clean01, out01, mask=None) -> float:
    """Headline: Gaussian-derivative (sigma 1 px) gradient magnitude, mean absolute difference.
    Both inputs quantized to 8-bit first."""
    g1, g2 = grad_mag_gauss(_q(clean01)), grad_mag_gauss(_q(out01))
    d = np.abs(g1 - g2)
    return float(d.mean() if mask is None else d[mask].mean())

def sobel_grad_mag(img01):
    """Same operator as the G loss: Sobel kernels divided by 8, replicate-style border (mode='nearest')."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    gx = ndi.correlate(img01.astype(np.float64), kx, mode="nearest")
    gy = ndi.correlate(img01.astype(np.float64), kx.T, mode="nearest")
    return np.hypot(gx, gy)

def gradient_error_sobel(clean01, out01, mask=None) -> float:
    d = np.abs(sobel_grad_mag(_q(clean01)) - sobel_grad_mag(_q(out01)))
    return float(d.mean() if mask is None else d[mask].mean())
```

- `gradient_error` and `gradient_error_sobel` in the CSV are **global**. The edge-region versions go in extra columns (Section 10).
- Label every Sobel-based number **"same operator as the G loss"**; it is circular for G models. Never use it as a headline.
- Gradient error is a structural proxy, not a perceptual metric.
- Optional Canny agreement is called "detector agreement" (precision/recall/F1 vs edges from the clean image, small tolerance, strict one-to-one matching, fixed and tuned thresholds). Not core.

---

## 10. Results CSV (`artifacts/results/results.csv`) , one row per image x sigma x method x loss variant x train seed

**Required columns, in this exact order:**

```
image_id, split, seed, sigma_255, method, parameter_config, model_id_optional, loss_variant, train_seed, psnr, ssim, smooth_psnr, edge_psnr, gradient_error, gradient_error_sobel, edge_f1_optional, runtime_seconds, hardware
```

**Extra columns [PROPOSED]**, appended **after** `hardware`, never inserted before:

```
texture_psnr, smooth_mae, edge_mae, gradient_error_edge, gradient_error_sobel_edge, n_smooth_px, n_edge_px, n_texture_px, warnings
```

(The plan asks for gradient error "globally and inside edge regions", but the base field list only has global columns, hence the appended edge columns.)

Field semantics:

| Field | Value |
|---|---|
| `image_id` | manifest ID |
| `split` | `test` for final results (`calibration` for tuning logs, `validation` for DL checkpoint/pilot logs) |
| `seed` | **noise** seed identity: `GLOBAL_SEED` (the per-image seeds derive from it), as an int |
| `sigma_255` | 15, 25, 50 |
| `method` | `noisy`, `gaussian`, `bilateral`, `tv`, `nlm`, `dncnn`, `unet` |
| `parameter_config` | classical: `config_id(config)` of the **frozen** setting for that sigma. DL: checkpoint config id |
| `model_id_optional` | DL only, e.g. `unet_G_s1`; empty for classical |
| `loss_variant` | `P` or `G` for DL; `none` for classical and `noisy` |
| `train_seed` | 0 or 1 for DL; empty for classical |
| `psnr`, `ssim` | full image, quantized |
| `smooth_psnr`, `edge_psnr` | regional PSNR with masks from Section 9 |
| `gradient_error` | Gaussian-derivative headline, global |
| `gradient_error_sobel` | "same operator as the G loss", global |
| `edge_f1_optional` | empty unless Canny agreement is run |
| `runtime_seconds` | per-image wall time under the harness conventions of Section 12 |
| `hardware` | string from `common/versions.py`, e.g. `cpu:<model>|gpu:none` or `gpu:<name>` |

- Missing values are empty fields, not `0` or `-1`. Floats written with `repr` precision (no rounding).
- **Primary key** for duplicate detection: `(image_id, sigma_255, method, parameter_config, loss_variant, train_seed)`.
- A and B write results through the **same** `evaluate/evaluate.py`; A supplies runtime numbers via `artifacts/results/runtime.csv`, merged on the primary key. **[PROPOSED]**
- Main tables use **per-image values**; for DL models average the two train seeds per image for the headline, and report the two seeds separately for spread. **[PROPOSED]**

### 10.1 Integrity checks (`evaluate/integrity.py`, run automatically, fail loudly)

- Output same shape and scale as the clean reference; finite; within [0,1] before quantizing.
- Input array unchanged after the call (hash before/after); no in-place changes.
- Correct references: clean image ID matches noisy ID; noisy PNG hash matches the one regenerated from the seed (spot check).
- Row count equals expected: `60 images x 3 sigmas x (n methods and variants and seeds)`. No missing and no duplicate primary keys.
- Noisy-input PSNR is near 24.6 / 20.2 / 14.2 dB at sigma 15 / 25 / 50 (a bit higher due to clipping).
- Failures and warnings (e.g. TV convergence, NaN mask, capped PSNR) are logged to `artifacts/logs/` **and** written to the `warnings` column; failed runs are never dropped silently.
- Test images were not used in `tuning/` or in DL validation logs (cross-check against manifest splits).

---

## 11. Classical methods: tuning protocol

- Tune **per sigma** on the **8 calibration images only**, with a **small grid declared in advance** in `configs/{method}_grid.yaml`. Objective: **mean quantized PSNR** over the 8 images at that sigma. Tie-break: the first config in the declared grid order. **[PROPOSED]**
- Save the full grid results (`artifacts/tuning/{method}_sigma{S}_grid.csv`) and the chosen config (`artifacts/tuning/{method}_frozen.json`, keyed by sigma). **Frozen settings are final; no retuning after seeing test results.**
- Reuse the existing `filters/` and `tune.py` (Gaussian, bilateral, median, Perona-Malik) behind the interface; extend the search spaces for NLM and TV.
- **Smoke test before the full run:** a few images, all methods, every integrity check green.

### 11.1 Method specifics

**Gaussian (A).** Linear filter, `scipy.ndimage.gaussian_filter`-equivalent or the existing from-scratch code. Config: `sigma_px`, border mode (`reflect`), kernel truncation. Grid declared in advance.

**Bilateral (B).** Local weighted average with weights = spatial Gaussian x range (intensity-similarity) Gaussian, normalized per pixel. Config: `sigma_s_px` (spatial), `sigma_r_255` (range, 0 to 255 scale, converted to /255 inside), window radius / neighbourhood support, border handling (`reflect`). Document implementation and version. Expected failures: weak-contrast boundaries, noisy similarity weights, texture flattening, runtime.

**TV (B).** State the exact objective and convention (isotropic vs anisotropic), solver, regularization parameter, stopping criterion, iteration limit. Library OK if labelled, e.g. scikit-image `denoise_tv_chambolle(image, weight, eps, max_num_iter)`. Read the installed version's docstring and record the exact objective and weight convention in the module docstring. **Never mix solvers:** `denoise_tv_bregman` uses a different weight convention. Config: `weight`, `eps`, `max_num_iter`. Log convergence warnings. Expected failures: texture flattening, staircasing.

**NLM (A).** Non-local patch similarity (library OK if labelled, e.g. scikit-image `denoise_nl_means` or OpenCV). Config: `patch_size`, `patch_distance`, `h` (the main smoothing parameter on the [0,1] scale), `sigma` estimate if passed (allowed: classical methods are told sigma), `fast_mode`. Expected cost: runtime; weakness: rare patterns, severe noise.

---

## 12. Runtime and hardware (A owns the harness)

- Report CPU/GPU, framework and software versions. Warm up learned-model inference. Time repeated runs and report a robust summary (median, plus IQR). State whether model loading and data transfer are included (**[PROPOSED]**: model loading excluded, host-to-device transfer included for GPU).
- Run **all** methods on the **same CPU**, including DL models, and **additionally** report DL on GPU. Never compare one method's GPU time against another's CPU time as if the hardware were equivalent. If DL ran on GPU and classical on CPU, keep quality and timing conclusions separate.
- `runtime_seconds` in the main CSV is per image, per method, with the hardware string stored beside it.

---

## 13. Fair-comparison statement (must accompany every comparison table or plot)

> Classical methods are tuned per sigma on the calibration images and are told sigma; the deep learning models are blind and were trained on sigma in [15, 50] (test sigma 15 and 50 lie at the ends of the training range).

Also state: both networks were trained with the same recipe, budget, checkpoint rule and noise sampler; differences cannot be attributed to architecture type alone.

---

## 14. Deep learning: shared recipe

Both networks use the **same trainer** (`common/dl_train.py`), the same data pipeline, the same noise sampler, the same checkpoint rule. Only `models/*.py` differ.

### 14.1 Fixed by the frozen protocol

- BSD400 split **before** patch extraction: 380 train / 20 validation.
- **64 x 64** patches; augmentation = random horizontal/vertical flips and 90-degree rotations (applied to the clean patch before noising).
- Online AWGN, sigma ~ U[15, 50] per patch, noisy-image rule identical to test (Section 7.3). **Blind** models.
- Optimizer **Adam**; base pixel loss **L1**; compact networks; parameter counts reported (keep U-Net in the same range as DnCNN).
- 2 training seeds (0, 1) per (network x loss variant).
- Checkpoint and lambda choices use **only the 20 validation images**.

### 14.2 Not fixed anywhere , decide together on Day 2 **[OPEN; PROPOSED defaults]**

| Setting | Proposed default |
|---|---|
| Batch size | 64 |
| Steps (identical for both nets) | fixed budget, e.g. 30 000 steps (shrink transparently if Kaggle time requires) |
| LR schedule | Adam lr 1e-3, halve at fixed step milestones (or cosine to 1e-5); same for both |
| Validation | every 1 000 steps, on the 20 validation images, full-size, fixed noisy realizations at sigma 15, 25, 50 (generated once with the Section 7.2 generator, saved to `artifacts/val_noisy/`) |
| Checkpoint rule | best mean quantized PSNR over the 3 sigmas on the validation images; keep `best.pt` and `last.pt` |
| Mixed precision | off for the main runs (document if on) |
| Gradient clipping | none unless instability appears (then for both) |
| Checkpoint selection never uses | any BSD68 image |

### 14.3 Model conventions

- Both predict the **noise residual** with a one-channel head; the restored image is `restored = noisy - predicted_noise`. Final output is clipped to [0,1] only at inference/evaluation, **not inside the training loss**.
- **DnCNN (A):** document depth, width, normalization, receptive field, parameter count.
- **Compact residual U-Net (B):** encoder-decoder, skip connections (e.g. 3 levels), documented channel widths, normalization, parameter count; one-channel residual head. Input H and W may not be divisible by the downsampling factor: **pad (reflect) to a multiple of `2**levels`, run, crop back, and record this** in the wrapper docstring and `meta.json`. DnCNN is fully convolutional and needs no padding (`pad_multiple = 1`).
- Differences from DnCNN (depth, skips, normalization) mean results must not be attributed to architecture type alone.

### 14.4 Loss (`common/losses.py`) , reference implementation

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SobelGradLoss(nn.Module):
    """L1 between Sobel-x and Sobel-y responses of pred and target. Kernels divided by 8.
    Border: replicate padding, then 'valid' conv (matches scipy mode='nearest')."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def _grads(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.kx), F.conv2d(x, self.ky)

    def forward(self, pred, target):
        gx_p, gy_p = self._grads(pred)
        gx_t, gy_t = self._grads(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

def total_loss(restored, clean, variant: str, lam: float, sobel: SobelGradLoss):
    """restored = noisy - predicted_noise (unclipped). variant 'P' -> L1 only; 'G' -> L1 + lam * Sobel L1."""
    pix = F.l1_loss(restored, clean)
    if variant == "P":
        return pix
    if variant == "G":
        return pix + lam * sobel(restored, clean)
    raise ValueError(variant)
```

- The two Sobel terms are each **mean-reduced L1**, summed (not averaged). `lam` multiplies their sum.
- **Lambda pilot:** G with each of 0.1, 0.5, 1.0, on validation images only, a short run; choose the value with the best validation criterion (the same criterion as the checkpoint rule; state it). Then set `LAMBDA_FROZEN` **once, used by both networks** **[PROPOSED]** (frozen protocol says "fixed", this keeps the ablation consistent; if each person pilots separately, record both values and state it). Record the pilot table in `docs/lambda_pilot.md`.

### 14.5 Loss unit tests (required)

- Constant image -> Sobel loss 0.
- `pred == target` -> total loss 0 for P and G.
- Horizontal ramp `x[i, j] = j`: in the interior `gx = 2*(1+2+1)/8 = 1.0` (per unit slope) and `gy = 0`; a ramp scaled by `c` gives `gx = c`.
- `lam = 0` makes G identical to P (value and gradient).
- Gradients are finite; sign and symmetry (swapping `pred`/`target` gives the same loss).
- Shape-agnostic for `(B,1,H,W)` with odd sizes.

### 14.6 DL acceptance tests (both networks)

- Overfit sanity: a tiny subset reaches near-zero loss.
- Residual sign correct: with the trained model on validation images, `restored` PSNR > noisy PSNR at all 3 sigmas.
- **A working DL model should clearly beat the best classical result on validation images**; if not, suspect a bug (scale, residual sign, normalization mode, too few steps). Do not tune toward published numbers on test data.
- Noisy-input rule check: statistics of training noisy patches match the saved test files (same quantized-grid property; pre-clip mean/std).
- Output shape = input shape for odd/non-multiple sizes (U-Net).
- No sigma anywhere in `denoise_dl` or the model's `forward`.
- Same patch stream (data generator) in P and G for the same `train_seed` (compare a hash of the first N batch indices).

### 14.7 DL wrapper (reference, `methods/unet.py` and `methods/dncnn.py`)

```python
_CACHE: dict = {}

def denoise_dl(image: np.ndarray, model_config: dict) -> np.ndarray:
    """Blind. model_config: {'checkpoint': path, 'device': 'cpu'|'cuda', 'arch': 'unet'|'dncnn', 'pad_multiple': int}"""
    check_image(image)
    model = _load(model_config)                     # cached by (checkpoint, device); eval(), no_grad
    m = model_config.get("pad_multiple", 1)
    x = torch.from_numpy(image.copy())[None, None].to(model_config["device"])
    H, W = x.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        residual = model(x)
        restored = x - residual
    restored = restored[..., :H, :W]               # crop back
    out = restored.squeeze().float().cpu().numpy()
    return np.clip(out, 0.0, 1.0).astype(np.float32)
```

### 14.8 Checkpoint provenance (`artifacts/checkpoints/{model_id}/meta.json`)

`model_id` (`{arch}_{variant}_s{train_seed}`), architecture and variant, source paper, parameter count, training noise range, input normalization, blind/sigma-specific, framework + version, hardware, precision, steps, batch size, LR schedule, seeds (init/data/noise), lambda, checkpoint rule and chosen step, validation PSNR curve file, git commit, checkpoint SHA-256, and any mismatch between training assumptions and the test protocol.

---

## 15. Analyses

### 15.1 Statistics (A, `common/stats.py`) , B calls these, does not reimplement

```python
def paired_diff(df, method_a, method_b, metric, sigma, **filters) -> np.ndarray:   # per image, a - b
def bootstrap_ci(diffs: np.ndarray, n_boot=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA, seed=GLOBAL_SEED) -> tuple[float, float, float]:
    """Resample IMAGES with replacement; return (mean, lo, hi) percentile CI."""
def seed_spread(df, model_prefix, metric, sigma) -> dict:   # mean, min, max, std across train seeds
```

- **Images are the experimental unit.** Paired per-image differences, bootstrap 95% CI over images. Report seed spread for DL models, and judge P-vs-G effects **against seed spread**.
- Conclusions rest on per-image outcomes; check stability across images (not dominated by a few).

### 15.2 Region analysis (B)

Per sigma, per method: smooth/edge (and optional texture) PSNR or MAE; gradient error global and in edge regions; pixel counts; sensitivity to percentile moves. Do overall and regional rankings differ?

### 15.3 Equal-smoothing comparison (B, `analysis/equal_smoothing.py`)

1. **Declare in advance** (`configs/equal_smoothing.yaml`, committed before running) a sweep grid for the main smoothing parameter of each classical method: Gaussian `sigma_px`, bilateral `sigma_s_px` **or** `sigma_r_255` (state which), TV `weight`, NLM `h`; other parameters at calibrated values. Per sigma.
2. For each setting and sigma, averaged over the **60 test images**: **smooth-region PSNR** (noise removed) and **edge-region PSNR and gradient error** (structure kept).
3. Plot **edge-region quality (y) vs smooth-region PSNR (x)**: one curve per classical method; one point per trained network (**P and G separate**, seed-averaged with seed spread shown).
4. Read each method's edge metric at **matched smooth-region PSNR** (e.g. the Gaussian's calibrated level), by linear interpolation along the curve; **no extrapolation** (report NA if outside the curve's range). Report who is above whom, with image-level uncertainty from A's bootstrap code.
5. This is **descriptive analysis on the test set with a pre-declared grid; it tunes nothing.**

Output `artifacts/results/equal_smoothing.csv` columns **[PROPOSED]**: `sigma_255, method, setting_name, setting_value, smooth_psnr_mean, edge_psnr_mean, gradient_error_mean, gradient_error_edge_mean, n_images`.

### 15.4 P versus G ablation (both networks)

Per image: PSNR, SSIM, edge-region PSNR, gradient error (Gaussian headline; Sobel only labelled). Compare against **seed spread**; ask whether the effect holds for both networks and what G costs in smooth regions.

### 15.5 Failure gallery (B)

Select examples by **defined criteria or systematic ranking** (largest per-image gap between methods; lowest edge-region score), **not only visual wins**. Cover thin lines, foliage or hair, weak edges, repeated texture, smooth gradients. Save the selection rule and the chosen image IDs. Relate each failure to the method's assumption.

---

## 16. Code style and engineering conventions

- **Python 3.10+**, type hints on all public functions, NumPy-style docstrings (units, shapes, value ranges stated).
- Libraries: numpy, scipy, scikit-image, Pillow, pandas, matplotlib, PyYAML, PyTorch, pytest. Pin exact versions in `requirements.txt`; record versions in every result (`common/versions.py`).
- **Format/lint:** ruff + black-compatible, line length 100. `pathlib.Path` for paths; **no hard-coded absolute paths**; every path comes from a config or CLI argument.
- Arrays: `float32` for images in the interface; `float64` internally for metrics. Shape `(H, W)` for images, `(B, 1, H, W)` for torch batches.
- **No global mutable state** (except the read-only model cache in the DL wrapper). No `print` in library code, use `logging`.
- **Randomness:** only `np.random.default_rng(SeedSequence([...]))` and `torch.Generator`; never `np.random.seed`, never `hash()`.
- CLI scripts use `argparse`, take `--config`, `--out`, `--seed` where relevant, and are idempotent (skip existing outputs unless `--force`).
- Every produced artifact has a producing script; one-command reproduction through `make all` (or `python -m scripts.run_all`), stages: `manifest -> noisy -> masks -> tune -> train -> infer -> evaluate -> integrity -> analysis -> plots`.
- **Plot style (`common/plotting.py`):** one fixed color and marker per method, used by both people; colorblind-safe palette; axis labels with units (dB, px, s); error bars/bands from bootstrap or seed spread; save PNG (200 dpi) and PDF to `artifacts/figures/`; the fairness statement (Section 13) goes in the caption or footnote of comparison figures; a single seeded RNG for any jitter.
- Naming: methods lowercase (`unet`), loss variants `P`/`G`, model ids `unet_G_s1`, sigma directories `sigma25`, figures `plot5_regionwise.png`.
- Commit messages: `area: short description`. Never commit data or checkpoints; commit configs, frozen JSONs, manifest, and small results.

---

## 17. Required tests (`tests/`, pytest; run in CI or before every handoff)

| Area | Test |
|---|---|
| I/O | load returns float32 (H,W) in [0,1]; PNG round-trip is exact for 8-bit levels |
| Noise | deterministic: same (index, sigma) gives identical output across runs; pre-clip mean/std within tolerance; output on the 8-bit grid; saved PNG equals regenerated array |
| Manifest | counts 8/60/380/20; no overlap; hashes valid; BSD68 not in BSD400 |
| Interface | every method: shape preserved, dtype float32, finite, in [0,1], input not modified, deterministic |
| Metrics | PSNR of identical images = capped value; PSNR of known MSE analytic value; SSIM of identical = 1; SSIM validated against a second reference; noisy PSNR near 24.6/20.2/14.2 |
| Regions | thresholds fitted on calibration only (assert IDs); masks boolean, correct shape; pixel counts nonzero; smooth and edge disjoint; synthetic step edge lies in `edge`; flat patch lies in `smooth` |
| Gradient error | zero for identical; known value for synthetic blur |
| Loss | Section 14.5 |
| Models | param count matches doc; forward shape; U-Net odd-size pad/crop; no sigma argument |
| DL wrapper | padded vs unpadded consistency for already-multiple sizes; output clipped |
| Evaluator | CSV column order exact; primary-key uniqueness; integrity failures raise |
| Equal smoothing | interpolation returns NA outside range; no test-set tuning |

---

## 18. Sanity yardsticks and reproducibility

- Noisy-input PSNR near the theoretical **24.6, 20.2, 14.2 dB** at sigma 15, 25, 50 (slightly higher due to clipping).
- Existing tuning on 8 calibration crops gave roughly **29.3, 26.3, 22.6 dB** for the best classical method at sigma 15, 25, 50. A working DL model should clearly beat classical results on validation images.
- For context only (not a target): published DnCNN on BSD68 is about 31.6, 29.2, 26.2 dB (Restormer paper, Table 4). A compact, briefly trained network will probably be lower. **Do not tune toward these on test data.**
- Save with every run: seeds, preprocessing, configs, checkpoint IDs/hashes, software versions, hardware, manifest, logs.

### Checklist (both)

- [ ] Test set and splits frozen before tuning; BSD68 never used for training or model selection.
- [ ] Same noisy realization across methods; seeds and preprocessing recorded; noisy-image rule identical in training and test.
- [ ] Classical search ranges and chosen settings saved.
- [ ] Both networks: architecture, parameters, steps, seeds, checkpoint rule, training curves documented.
- [ ] Lambda chosen on validation only; P and G matched in everything else.
- [ ] Runtime conditions stated for every implementation.
- [ ] Correct references, metric range and image scale verified.
- [ ] Failed runs and exclusions logged; conclusions based on per-image outcomes.

---

## 19. Schedule and handoffs

| Days | Work |
|---|---|
| 1-2 | Freeze scope: splits, manifest, protocol, metrics, masks, DL budget, lambda grid; read the primary paper; verify Kaggle GPU access |
| 3-5 | A: Gaussian, NLM, noise pipeline, DnCNN code + first P run. B: bilateral, TV, evaluator prototype, U-Net code + first P run. Both: sanity tests |
| 6-7 | Integration; calibrate classical methods per sigma and **freeze**; lambda pilots on validation; start G and second seeds |
| 8-10 | Main runs on the 60 test images at all sigmas; save outputs, per-image CSV, configs, logs; check for missing runs |
| 11-12 | Analysis: paired comparisons, region analysis, equal-smoothing curves, P vs G, runtime, parameter-response curves, failure gallery |
| 13-14 | Conclusions, limitations, slides, README, rehearsal (10-minute talk) |
| Optional week 3 | One extension only, after the core is complete |

| When | B gives A | A gives B |
|---|---|---|
| Day 2 | Agreement on masks and metric conventions | Manifest, splits, noise generator, frozen protocol |
| Day 5 | PSNR and SSIM functions, bilateral and TV wrappers | Gaussian and NLM wrappers, noisy test sets |
| Day 7 | Frozen bilateral/TV settings, evaluator CSV format, U-Net P checkpoint | Frozen Gaussian/NLM settings, DnCNN P checkpoint |
| Days 8-10 | Full evaluation CSV, masks, regional metrics, U-Net P and G outputs | DnCNN P and G outputs, runtime numbers |
| Days 11-12 | Equal-smoothing curves, failure gallery, plots 1, 5, 7 | Statistics, plots 2-4, parameter-response curves |

**If we slip, cut in this order:** optional items; the second seed of G; Canny; the regional texture class; NLM parameter grid size. **Never cut:** the six-method comparison, region-aware metrics, paired statistics, P versus G for at least one network.

---

## 20. Optional extensions (only after the core is complete; at most one or two)

Perona-Malik diffusion (7th method; already implemented in the course), median filter + impulse noise test, Restormer reference (official weights or from-scratch script), Canny agreement, frequency-band/Fourier analysis (high frequencies contain both detail and noise; Fourier magnitude discards phase), lambda sweep beyond the 3-point grid, sigma-specific DL models, Wiener **or** BM3D (not both), ZNCC only if defined consistently.

---

## 21. Open decisions to settle on Day 2 (then move to FROZEN and log in Section 22)

| # | Decision | Proposed |
|---|---|---|
| 1 | `GLOBAL_SEED` value | 20260 |
| 2 | How the 8 calibration and 20 validation images are drawn | seeded random draw, recorded in manifest |
| 3 | DL batch size, steps, LR schedule, validation frequency | Section 14.2 |
| 4 | Checkpoint criterion | best mean quantized PSNR over 3 sigmas on the 20 validation images |
| 5 | One shared lambda or one per network | one shared `LAMBDA_FROZEN` |
| 6 | Region threshold pooling | pooled pixels over the 8 calibration images |
| 7 | Edge-percentile sensitivity upper end | 99 instead of 100 |
| 8 | Extra CSV columns appended after `hardware` | Section 10 |
| 9 | DL headline aggregation across the 2 train seeds | per-image mean over seeds; seeds also shown separately |
| 10 | `MIN_MASK_PIXELS` | 500 |
| 11 | Runtime inclusion rules (model loading, transfer) | loading excluded, transfer included |
| 12 | Bilateral equal-smoothing axis | state spatial vs range |
| 13 | Plot numbering for A's parameter-response curves | plot 6 |
| 14 | Bootstrap resamples | 10 000 |
| 15 | Result storage of method outputs | quantized 8-bit PNG |

---

## 22. Changelog (edit together; newest first)

| Date | Change | Who agreed |
|---|---|---|
| (today) | Initial version generated from Project Plan + Person B guide | , |
| (today) | Repo scaffold built; manifest generated from real BSD68/BSD400; Section 25 added (file map, noise seed includes dataset code, rising-branch rule, TV/bilateral/NLM grids widened on calibration data) | , |

---

## 23. Background reference (from the plan and Person B's guide)

### Core hypotheses (test, do not assume)
- Gaussian may suppress noise while blurring fine structures.
- Bilateral may retain strong edges better but struggle with weak boundaries and texture.
- TV may retain prominent contours but produce staircasing or flatten texture.
- NLM may benefit from repeated patches but cost more runtime and struggle with rare patterns or severe noise.
- DnCNN and the U-Net may offer a strong quality/runtime trade-off when training assumptions match test noise, but can show domain mismatch (especially at the ends of the training range) or over-smoothing/artifacts.
- The Sobel loss (G) may improve edge-region and gradient metrics, may cost PSNR or amplify noise in smooth regions, or may make no measurable difference.

### Interpretation questions
- Best method per noise level; do rankings change with sigma? Do overall and regional rankings differ?
- At equal smoothing, which classical method keeps the most structure, and where do the networks sit?
- Does G help edge regions, what does it cost in smooth regions, is the effect larger than seed spread, for both networks?
- Which failures appear, and which algorithmic assumptions explain them?
- Quality/runtime trade-off under the stated hardware?
- Do the networks behave differently at sigma 15 and 50 (ends of the training range) than at 25?
- Stable across images or dominated by a few?
- What can grayscale AWGN reveal and not reveal about real photographs?

### Theory notes required
- **A (half page):** linear-filter variance, NLM weights, residual learning, receptive field.
- **B (half page):** bilateral weights (spatial x range Gaussian, normalized per pixel); the TV objective (data term + lambda x total variation), isotropic vs anisotropic, why staircasing appears; why skip connections help preserve detail in an encoder-decoder; why measuring Sobel error on a Sobel-trained model is circular.

### Paper adaptation card
Inspiration: Gingichashvili and Lischinski, *Evaluation and Comparison of Edge-Preserving Filters* (2020), https://arxiv.org/abs/2012.13778. It is about smoothing operators in computational photography, not denoising as such.
- **Borrow:** systematic comparison, comparison at an equivalent degree of smoothing, structural and edge-aware evaluation.
- **Adapt:** controlled noisy-photograph restoration, quality metrics, regional analysis, runtime.
- **Extend:** trained DnCNN and compact residual U-Net denoisers; the P-vs-G ablation.
- **Change:** selected methods, noise protocol, dataset, objectives.
- **Do not claim** exact reproduction or direct comparability to the paper's results.

### Course alignment
| Expectation | Where covered |
|---|---|
| Read papers | Edge-preserving filter evaluation paper; DnCNN; U-Net (architectural adaptation); NLM, TV, bilateral sources |
| Mathematics | Theory notes by each person |
| Implement, ideally from scratch | Gaussian and bilateral from scratch (existing code); DnCNN and U-Net written by us in PyTorch; NLM and TV may use a library, labelled |
| Datasets | BSD400 (training), BSD68 (test), masks and manifest as curated artifacts |
| Empirical analysis and ablations | Main comparison, equal-smoothing curves, parameter-response curves, P-vs-G ablation with seeds |
| Summarize and explain | 10-minute talk, README, report |

### Ten-minute talk
1. Application, motivation, question (1 min). 2. Dataset, noise, fair protocol (1.5). 3. Classical vs learned intuition, Sobel loss idea (2). 4. Main quality, regional and runtime results, equal-smoothing curves (2.5). 5. P-vs-G ablation, failure cases, interpretation (2). 6. Limitations and conclusion (1).

**Person B's presentation role:** explain bilateral weights and TV regularization; lead through metric limitations, regional findings, equal-smoothing curves, failure examples; help answer why a learned model might outperform or underperform classical methods, what the Sobel loss did and did not change, and what the experiment cannot establish.

### Moodle deliverables
Adaptation card; dataset manifest; code with one-command reproduction; model provenance; configs; per-image CSV; plots; failure gallery; README; report and slides.

### Person B deliverables checklist
- [ ] Bilateral and TV wrappers, grids, frozen settings
- [ ] U-Net code, configs, curves, checkpoints, P and G with 2 seeds; loss tests; lambda pilot record
- [ ] Evaluator core, CSV, integrity checks
- [ ] Masks, thresholds, pixel counts, sensitivity analysis; regional and gradient-error results
- [ ] Equal-smoothing curves and table
- [ ] Failure gallery, plots 1, 5, 7
- [ ] Theory note

### Person A deliverables (from the plan's division of work)
- [ ] Paper adaptation note
- [ ] Noise pipeline, manifest, seeds, I/O tests
- [ ] Gaussian (reuse) and NLM wrappers, frozen settings
- [ ] DnCNN code and runs (P and G, 2 seeds each)
- [ ] Runtime benchmarking harness (all methods)
- [ ] Statistics: paired differences, bootstrap CIs, seed spread, P-vs-G tests
- [ ] Plots 2 to 4 and parameter-response curves
- [ ] Theory notes A

---

## 24. Kaggle execution notes (all heavy compute runs on Kaggle Notebooks)

Facts to design around (check your own account; Kaggle does not guarantee them): free T4 or P100 GPU (about 16 GB), weekly GPU quota commonly about 30 hours that resets weekly, session limit up to 12 hours (plan for **9 hours or less**), about 20 GB disk, internet is a per-notebook toggle. Each person uses their **own account and quota**. Running two GPU notebooks at once drains the same quota twice as fast, so run one GPU notebook at a time.

### 24.1 Paths: never hard-code, use `common/paths.py`

```python
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

def roots() -> dict:
    if ON_KAGGLE:
        inp = Path("/kaggle/input")
        return {
            "code": inp / "denoise-code",            # code dataset (or a git clone in /kaggle/working)
            "data": inp / "bsd-data",                # BSD400 + BSD68 dataset
            "shared": inp / "denoise-shared",        # manifest, noisy PNGs, frozen settings, checkpoints
            "work": Path("/kaggle/working"),         # the ONLY directory kept as notebook output
        }
    base = Path(os.environ.get("DENOISE_ROOT", ".")).resolve()
    return {"code": base, "data": base / "data", "shared": base / "artifacts", "work": base / "artifacts"}
```

- `/kaggle/input` is **read-only**. Everything a notebook writes goes to `roots()["work"]`, laid out like `artifacts/` in Section 4.
- Dataset slugs above are placeholders; set the real ones in `configs/kaggle.yaml`, never in code.

### 24.2 What lives where

| Item | Where | How it reaches the other person |
|---|---|---|
| Code + this file | private Kaggle dataset `denoise-code` (or GitHub clone if internet is on) | new dataset version after each merge |
| BSD400, BSD68 raw images | private dataset `bsd-data` (A uploads once, shares with B) | shared dataset |
| Manifest, noisy PNGs (calibration, test, validation), `thresholds.json` | dataset `denoise-shared`, built by **A**, then frozen | B adds it as notebook input; always pin the **version number** |
| Frozen classical settings, U-Net and DnCNN checkpoints, output PNGs, CSVs | each person publishes their own dataset version (`denoise-A-results`, `denoise-B-results`) | the other person pins that version for handoffs (Section 19) |

- Always **reference a specific dataset version** in the notebook and record the version number in `meta.json` and the results log. "Latest" is not reproducible.
- Both people must evaluate against the **same** `denoise-shared` version (same manifest hash, same noisy PNGs). `evaluate/integrity.py` checks the manifest hash against the value stored in the results.

### 24.3 Environment

- Use the preinstalled PyTorch, NumPy, SciPy, scikit-image, pandas, matplotlib. Avoid `pip install` (it needs internet). If something is missing, add it as a wheel inside the code dataset.
- First cell of every notebook: print and save `common/versions.py` output (Python, torch, CUDA, scikit-image, GPU name, CPU model) to `artifacts/logs/env_{timestamp}.json`.
- Notebooks are **thin**: they only set paths, import from the repo, and call a script/function. No logic lives only in a notebook, so the same code runs locally and satisfies the one-command-reproduction deliverable.

### 24.4 Resumable training (required)

A session can end (time limit, quota, crash). Every training run must resume from its last checkpoint.

```python
def save_ckpt(path, model, opt, step, gens, best, log_rows):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "gen_init": gens["init"].get_state(), "gen_data": gens["data"].get_state(),
                "gen_noise": gens["noise"].get_state(), "best": best, "log": log_rows}, path)

def try_resume(path, model, opt, gens):
    if not Path(path).exists():
        return 0, {"psnr": -1.0, "step": 0}, []
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    for k in ("init", "data", "noise"):
        gens[k].set_state(ck[f"gen_{k}"])
    return ck["step"], ck["best"], ck["log"]
```

- Save `last.pt` (with optimizer, **all three generator states**, step, best-so-far) every **1 000 steps** (same cadence as validation), and `best.pt` whenever the validation criterion improves (Section 14.2). Resuming must continue the same patch stream and noise draws, so a resumed run equals an uninterrupted run for P-vs-G pairing.
- Write logs by appending to `train_log.csv`, flush each write.
- A **stop guard**: the training script takes `--max_hours` (default 8.5); it checkpoints and exits cleanly before the session limit. Rerun the same command in a new session to continue.
- Test it: interrupt a short run at step N, resume, compare the final weights to an uninterrupted run (tolerance for GPU nondeterminism, exact equality on CPU).

### 24.5 Session and quota budget

| Job | Where | Budget |
|---|---|---|
| Manifest, noisy generation, masks | CPU notebook (A / B) | minutes |
| Classical tuning (calibration only) | CPU notebook | under 1 h each method; NLM is the slow one, keep its grid small |
| Lambda pilot (3 short G runs) | GPU | short, each run a fraction of a final run |
| Final DL runs: P and G x seeds 0, 1 | GPU, one run per session where possible | roughly 5 to 8 GPU hours per person in total (estimate; measure on the first P run and update this line) |
| Inference on 60 + 20 images x 3 sigmas | GPU and CPU | minutes |
| Full evaluation, equal-smoothing sweep, statistics, plots | CPU | under 1 to 2 h |

- Do a **measured throughput test** on day 3 to 5 (steps per second for each network on the assigned GPU) and write the real step budget into `configs/` before long runs. Both networks use the **same step count**; if one network is much slower, shrink the shared budget, do not give them different budgets.
- Keep GPU notebooks focused: no debugging on the GPU quota. Debug on CPU with a tiny subset, then switch the accelerator on.
- Turn off the GPU accelerator (or stop the session) as soon as a run finishes.

### 24.6 Commit workflow

Use "Save & Run All (Commit)" for final runs so outputs are stored as versioned notebook output; for long training, run the notebook interactively with the stop guard and then save the outputs as a dataset version. After each handoff milestone (Section 19): create a dataset version, note its version number in the changelog (Section 22), and tell the other person.

### 24.7 Runtime benchmarking on Kaggle (A)

- Kaggle CPUs are shared and can vary between sessions, so **run all methods in the same session, in an interleaved order**, with warm-up and several repeats; report the **median** and IQR. Record `lscpu` model, core count, thread settings (`torch.set_num_threads`, `OMP_NUM_THREADS`) and keep them identical for all methods.
- Report DL on **CPU** (comparable with classical) **and** on GPU (clearly labelled with GPU name). Never mix them in one comparison.
- State in the table that the numbers come from shared Kaggle hardware and include the session date.

### 24.8 Reproducibility caveats on Kaggle

- A session may give a T4 one time and a P100 another. Results are **not bit-identical across GPU types**. Record the GPU in `meta.json` and in the CSV `hardware` column; where possible train both seeds of a P/G pair on the same GPU type; do not claim exact bit reproducibility.
- Set `torch.backends.cudnn.benchmark = False` and `torch.use_deterministic_algorithms(True, warn_only=True)` for the final runs; note any nondeterministic ops warned about.
- Noise is drawn on CPU generators (Section 7.3), so the noisy data are identical on any hardware.

### 24.9 Kaggle checklist

- [ ] `common/paths.py` used everywhere; no `/kaggle/...` string elsewhere.
- [ ] Dataset versions pinned and recorded (code, data, shared, other person's results).
- [ ] Manifest hash identical in both people's results.
- [ ] Training resumes correctly; `--max_hours` guard in place.
- [ ] Real step budget measured and written to configs; same for both networks.
- [ ] Environment info saved at the start of every notebook.
- [ ] Final outputs saved as a dataset version and the version noted in the changelog.


---

## 25. Repo as built (authoritative file map and build decisions)

### 25.1 What is in the repo (everything else lives in Kaggle notebooks or Kaggle datasets)

```
common/    protocol.py paths.py io.py noise.py manifest.py metrics.py regions.py evaluator.py
           stats.py losses.py dl_infer.py dl_train.py registry.py versions.py
methods/   gaussian.py bilateral.py tv.py nlm.py dl_common.py dncnn.py unet.py
models/    __init__.py dncnn.py unet.py
scripts/   _common.py make_manifest.py make_noisy.py make_masks.py tune.py train.py infer.py evaluate.py
analysis/  equal_smoothing.py
configs/   train/train.yaml  grids/{gaussian,bilateral,tv,nlm}.yaml  equal_smoothing.yaml
tests/     conftest.py test_noise test_manifest test_metrics test_regions test_methods
           test_evaluator_stats test_torch_parts
docs/      kaggle_setup.md reference_run.md adaptation_card.md lambda_pilot.md model_provenance.md theory_A.md theory_B.md
artifacts/ manifest.json (committed, FROZEN)  masks/thresholds.json + pixel_counts.csv (committed)
           noisy/ outputs/ checkpoints/ tuning/ results/ logs/ figures/   (generated; big binaries git-ignored)
notebooks/ person_A/ person_B/  (optional thin notebook exports)
```

**Notebook-only (deliberately not in the repo):** all plots (1 to 7), the failure gallery, the runtime benchmarking
harness, exploratory analysis. Each person writes these in their own Kaggle notebook, calling the library above.
`common/plotting.py` from Section 4 was dropped; agree the per-method colour map inside the notebooks (document it in
`docs/`) or add the file later by agreement.

**Mapping from earlier sections:** `evaluate/evaluate.py` + `integrity.py` -> `common/evaluator.py` + `scripts/evaluate.py`;
`common/dl_train.py` also contains the sampler/validation/checkpoint code; there is no `scripts/run_all` (use the stage table in
README.md); `common/stats.py` has `bootstrap_ci`, `paired_diff`, `seed_spread`, `add_method_id`.

### 25.2 Decisions made while building (all confirm-able; items 1 to 4 change earlier text)

1. **Noise seed includes the dataset:** `SeedSequence([global_seed, dataset_code, index, sigma])`, `dataset_code` BSD68=68, BSD400=400.
2. **`image_id`** is the file stem (`test001` in BSD68, `test_001` in BSD400); uniqueness is per `(dataset, image_id)`. The manifest check
   that BSD68 and BSD400 do not overlap is by file hash only; they differ in size (full images vs 180x180 crops), so a content-level
   overlap cannot be ruled out by hash. The standard splits are disjoint by construction (BSDS500 train/val vs test).
3. **Calibration/validation draw:** `default_rng(SeedSequence([global_seed, 1]))` picks 8 of 68 BSD68 images; `[global_seed, 2]` picks 20 of 400 BSD400 images.
   Current manifest (`GLOBAL_SEED = 20260`): calibration = test027, test043, test049, test054, test056, test059, test060, test065.
4. **Equal-smoothing matching uses the rising branch only** (settings up to the peak of mean smooth-region PSNR). Smooth-region PSNR peaks
   and then falls with over-smoothing, so the curve folds back and a matched reading would be ambiguous. Off-curve targets give NaN.
5. **Validation noise** is regenerated in memory by `make_noisy` with the Section 7.2 seeds (no files needed); `--with-val` also writes PNGs for reference.
6. **Checkpoint files:** `best.pt` (model, arch, arch_cfg, step, val) is what inference loads; `last.pt` (adds optimizer, three generator states, log) is for resume.
   `last.pt` is written atomically. Rerun the same `scripts.train` command to resume.
7. **U-Net:** 3 downsamplings, widths 16/32/64/128, GroupNorm, ConvTranspose, concat skips, 1x1 residual head, pad multiple 8, ~0.48M params
   (hand-computed 482 449; DnCNN 17x64 ~556 096). Confirm the real counts with the torch test on Kaggle and put them in `docs/model_provenance.md`.
8. **Grids:** initial TV, bilateral and NLM grids had optima on a grid edge and were widened using calibration images only; check this for any new grid
   (optimum must be interior). Final grids are in `configs/grids/`.
9. **Small regions:** some images have very small smooth or edge regions (BSD68 default percentiles: min smooth 870 px, min edge 1 972 px; in the
   low/high sensitivity variants a few images fall below `MIN_MASK_PIXELS` = 500, giving NaN + a `small_mask_*` warning). Regional averages must use only
   non-NaN images and report how many images contributed.
10. **Runtime:** measured by `scripts.infer` per image (GPU synchronised, model load excluded, transfer included). Treat these as raw timings; the
    careful interleaved benchmark with medians is A's notebook (Section 12, 24.7).
11. **DL training steps/batch/lr** in `configs/train/train.yaml` are placeholders (30 000 steps, batch 64, lr 1e-3 with two decays) until the Kaggle
    throughput test; both networks always share them.

### 25.3 Verification status of this scaffold
- Executed and passing: manifest generation and validation, noise generation, mask thresholds, tuning for all four classical methods, inference, evaluation
  with integrity checks (900 result rows, column order exact), equal-smoothing sweep, and 31 tests. See `docs/reference_run.md` for numbers to reproduce.
- **Not executed (torch unavailable where this was built):** `models/`, `common/losses.py`, `common/dl_infer.py`, `common/dl_train.py`, `methods/dl_common.py`,
  `scripts/train.py`, `scripts/infer.py` for DL methods, and the torch tests. First action on Kaggle: `python -m pytest`, then the `--bench` test.
