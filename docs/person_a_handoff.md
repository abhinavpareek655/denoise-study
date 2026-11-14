# Person A Handoff: Person B Status

**Updated:** 2026-11-14

## Work completed by Person B

- Bilateral and TV settings were tuned on the calibration images. The selected settings match the grid maxima and are interior to the searched ranges.
- Bilateral and TV inference was run for 60 BSD68 test images at each of sigma 15, 25, and 50. Runtime CSVs and frozen tuning files are in the repository; the PNGs were generated locally and are not committed.
- The U-Net has 482,449 parameters. B ran the three 3,000-step validation pilots and trained P and G for seeds 0 and 1 for 20,000 steps each. The saved B2 output reports all four inference runs completed for 60 test images at all three sigmas.
- B selected lambda 0.5 as the shared value based on A's recorded decision. B's pilot had its highest mean PSNR at 0.1, and the two contributors used different validation splits, so A should confirm the shared value before final reporting.

## Work for Person A to coordinate or complete

1. Confirm the shared lambda 0.5 with B and keep the agreement in the project changelog.
2. Provide or identify BSD68 clean images whose hashes match `artifacts/manifest.json`. A's currently available legacy BSD68 files do not match, so they cannot be used for final scoring.
3. Ensure B saves and publishes the Kaggle output as a `denoise-B-results` version. The Git repository contains P checkpoints, but B's final G checkpoint weights and inference PNGs are not committed; the saved Kaggle version is needed for handoff.
4. Complete A's Gaussian, NLM, and DnCNN outputs and publish them in the current artifact format. Then merge A's and B's outputs and run `python -m scripts.evaluate --strict` on the matching clean BSD68 data.
5. Run the remaining analysis: equal-smoothing results, plot 1, plot 5, plot 7/failure gallery, and final provenance/report updates.

B's current `denoise-shared.zip` is built from this repository's manifest, noisy test PNGs, masks, and B's frozen bilateral/TV settings. It is not A's legacy dataset.