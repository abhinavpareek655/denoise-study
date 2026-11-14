# Lambda pilot record (validation images only; grid 0.1, 0.5, 1.0)

Raw rows are written by `scripts.train --pilot` to `artifacts/tuning/lambda_pilot_{arch}.csv`.
Selection criterion = the checkpoint criterion (best mean quantized PSNR over sigma 15/25/50 on the 20
validation images); also note validation gradient error. Pilot steps: see `configs/train/train.yaml`.

| Arch | lambda | seed | val PSNR (mean) | val grad err | chosen? |
|---|---|---|---|---|---|
| | | | | | |

Chosen LAMBDA_FROZEN: ____  (set in `common/protocol.py`; same value for both networks; logged in the changelog)
