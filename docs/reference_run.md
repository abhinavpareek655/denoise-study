# Reference run (classical pipeline) - numbers to reproduce

Produced while building this scaffold: 1-core CPU sandbox, Python/numpy/scipy/scikit-image/pandas =
3.12.3 2.4.4 1.17.1 0.26.0 3.0.2. Mean over the 60 BSD68 test images, tuned per sigma on the 8 calibration images
with the grids in `configs/grids/`. Use these to check that your Kaggle run matches (small differences
across library versions are possible; investigate anything beyond ~0.05 dB).

| Method | PSNR s=15 | s=25 | s=50 |
|---|---|---|---|
| noisy | 24.80 | 20.48 | 14.92 |
| gaussian | 28.12 | 25.90 | 23.20 |
| bilateral | 29.57 | 26.72 | 23.39 |
| tv | 29.75 | 27.11 | 23.82 |
| nlm | 30.07 | 27.44 | 23.85 |

Frozen settings found: Gaussian sigma_px 0.75 / 1.0 / 1.5. Bilateral (sigma_s_px, sigma_r_255)
(1.5, 35) / (1.5, 75) / (2.0, 170). TV weight 0.04 / 0.08 / 0.15. NLM h 0.04 / 0.055 / 0.11.

Notes: (1) initial TV, bilateral and NLM grids had their optimum on a grid edge and were widened
(calibration images only) until the optimum was interior. Check this for any new grid. (2) Noisy PSNR
is a little above the 24.6 / 20.2 / 14.2 dB theory because clipping removes some noise energy. (3)
These runtimes (about 0.003 / 0.11 / 0.10 / 0.15 s per image) are NOT valid benchmarks.
