# Model provenance

One entry per trained model. Most fields are auto-written to `artifacts/checkpoints/{model_id}/meta.json`;
copy the final table here.

| model_id | architecture | variant | seed | params | steps | lambda | best step | val PSNR | GPU | checkpoint sha256 |
|---|---|---|---|---|---|---|---|---|---|---|

Architectures. DnCNN: Zhang et al. 2017 (arXiv:1608.03167), compact variant, depth 17, width 64, BatchNorm.
U-Net: Ronneberger et al. 2015 (arXiv:1505.04597), a biomedical-segmentation architecture adapted by us
for denoising (3 downsamplings, widths 16/32/64/128, GroupNorm, residual head); not evaluated for
denoising in the original paper. Both: blind, trained on sigma ~ U[15,50], BSD400 (380 train / 20 val).
Training/test mismatch to state: test sigma 15 and 50 sit at the ends of the training range.
