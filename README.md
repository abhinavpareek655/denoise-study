# denoise-study

An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness,
Detail Preservation, and Computational Efficiency. Six methods (Gaussian, bilateral, TV, NLM, DnCNN,
compact residual U-Net), BSD68 test set, AWGN sigma = 15 / 25 / 50, all heavy runs on **Kaggle**.

**Read `PROJECT_CONTEXT.md` first.** It is the engineering contract (frozen protocol, interfaces, CSV
format, seeds). Paste it into every LLM coding session. Section 25 there is the file map of this repo.

## Repo rules (Kaggle-first)
- The repo holds the **shared library, CLI stages, configs, tests, docs and small result files**.
- It does **not** hold datasets, noisy PNGs, model outputs or checkpoints (git-ignored; they live in
  Kaggle notebook output / Kaggle dataset versions).
- Anything specific to one person's workflow (plots, failure gallery, runtime harness, exploration)
  lives in that person's Kaggle notebook, exported into `notebooks/person_A/` or `notebooks/person_B/`
  if you want it versioned. Notebooks stay **thin**: set paths, call `scripts.*` / `common.*`.
- Never hard-code `/kaggle/...`; use `common/paths.py` (env vars `DENOISE_DATA`, `DENOISE_SHARED`).

## Kaggle quick start
1. Upload `bsd-data.zip` (BSD68/ and BSD400/ folders) as a private Kaggle dataset named `bsd-data`.
2. In a notebook: add the dataset as input, turn Internet on, then run the bootstrap cell from
   `docs/kaggle_setup.md` (clones this repo into `/kaggle/working/denoise-study`).
3. Run stages from the repo root (below). Everything is written to `artifacts/`, which Kaggle keeps
   as notebook output; publish it as a dataset version for handoffs and pin the version number.

## Stages (all `python -m ...` from the repo root)
| Stage | Command | Owner |
|---|---|---|
| Manifest (already committed, frozen) | `scripts.make_manifest` | A |
| Noisy test sets | `scripts.make_noisy` | A |
| Region thresholds (already committed) | `scripts.make_masks` | B |
| Tune classical (calibration only) | `scripts.tune --method gaussian\|bilateral\|tv\|nlm` | A: gaussian, nlm. B: bilateral, tv |
| Throughput test | `scripts.train --arch unet --variant P --seed 0 --bench 200` | A, B |
| Lambda pilot (validation only) | `scripts.train --arch unet --variant G --lam 0.5 --pilot` (0.1, 0.5, 1.0) | A (dncnn), B (unet) |
| Train | `scripts.train --arch unet --variant P --seed 0` (P/G x seeds 0,1) | A (dncnn), B (unet) |
| Inference | `scripts.infer --method tv` or `--method unet --model-id unet_P_s0` | each person |
| Evaluate | `scripts.evaluate [--method-ids ...] [--mask-variant low\|high] [--strict]` | B (merge A's outputs) |
| Equal smoothing | `analysis.equal_smoothing` | B |
| Tests | `python -m pytest` | both (torch tests run on Kaggle) |

After the lambda pilot, set `LAMBDA_FROZEN` in `common/protocol.py` once (both networks), commit, and
log it in `PROJECT_CONTEXT.md` Section 22. Before long runs, edit `configs/train/train.yaml` with the
measured step budget (same for both networks).

## Notebooks (thin, optional exports)
- `notebooks/person_B/B1_cpu_prep.ipynb` - CPU session: bootstrap, environment log, `pytest`, parameter counts, input/hash checks,
  mask reproducibility check, tune bilateral and TV (calibration only).
- `notebooks/person_B/B2_gpu_unet.ipynb` - single-GPU session: choose jobs with `RUN` (`tests`, `bench`, `pilot`, `train_<P|G>_s<0|1>`,
  `infer_...`); restores checkpoints from an attached `denoise-B-results` version so training resumes.
- Both clone this repo, set `DENOISE_DATA`, and call `scripts.*`; edit the config cell (repo URL, pinned commit) before running.
  Details and the order of work: `PROJECT_CONTEXT.md` Section 25.4.

## Status of this scaffold
- Manifest, region thresholds and pixel counts are generated from the real BSD68/BSD400 files.
- Classical pipeline (noise, tuning, inference, evaluation, equal smoothing) was run end to end and
  its tests pass; see `docs/reference_run.md` for the numbers to reproduce.
- Torch code (models, losses, trainer, DL wrappers, `scripts.train`) was written and statically
  checked but **not executed** where this scaffold was built (no torch there). First action on Kaggle:
  `python -m pytest` (runs the torch tests incl. resume test), then the `--bench` throughput test.
- Person B notebooks (B1, B2) are written but **not yet run on Kaggle**; their first run is the first execution of the torch code.
