# Lambda pilot record (validation images only; grid 0.1, 0.5, 1.0)

Person A's legacy protocol selected lambda 0.5, and A's final DnCNN runs use it. Its BSD400 validation split has no images in common with
this repository's manifest, so the two pilot tables are not directly paired. B adopts A's value as the shared setting per the project owner's
instruction; confirm it with A before final reporting.

Raw rows are written by `scripts.train --pilot` to `artifacts/tuning/lambda_pilot_{arch}.csv`.
Selection criterion = the checkpoint criterion (best mean quantized PSNR over sigma 15/25/50 on the 20
validation images); also note validation gradient error. Pilot steps: see `configs/train/train.yaml`.

| Arch | lambda | seed | val PSNR (mean) | val grad err | chosen? |
|---|---|---|---|---|---|
| U-Net | 0.1 | 0 | 27.4243 | 0.009840 | candidate |
| U-Net | 0.5 | 0 | 27.3815 | 0.009946 | adopted shared value from A |
| U-Net | 1.0 | 0 | 27.4180 | 0.009845 | candidate |

Chosen `LAMBDA_FROZEN`: **0.5**. B's mean PSNR alone was highest at 0.1; the imported shared choice is based on A's documented selection, not
on a claim that B's pilot independently selected 0.5.
