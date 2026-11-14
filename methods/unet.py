"""Residual U-Net wrapper (Person B). Blind; input is reflect-padded to a multiple of 8 and cropped back."""
from methods.dl_common import denoise_dl  # noqa: F401

METHOD = "unet"
ARCH = "unet"
