# Paper adaptation card

Inspiration: Gingichashvili and Lischinski, *Evaluation and Comparison of Edge-Preserving Filters* (2020),
https://arxiv.org/abs/2012.13778. It studies smoothing operators in computational photography, not denoising as such.

- **Borrow:** systematic comparison; comparison at an equivalent degree of smoothing; structural / edge-aware evaluation.
- **Adapt:** controlled noisy-photograph restoration; quality metrics; regional analysis; runtime.
- **Extend:** trained DnCNN and compact residual U-Net denoisers; the P-vs-G (Sobel gradient loss) ablation.
- **Change:** selected methods, noise protocol (grayscale AWGN), dataset (BSD68 test, BSD400 train), objectives.
- **Do not claim** exact reproduction of, or direct comparability to, the paper's results.
