"""Thermal-specific data augmentation for OGI (Optical Gas Imaging).

Unlike natural RGB images, thermal IR images have distinct statistics and
augmentation requirements. Standard color jitter/shift is irrelevant; instead,
we use geometric + intensity transforms that preserve thermal semantics.

Transforms (applied identically to image and mask):
  - Random horizontal flip (p=0.5)
  - Random affine (rotation ±15°, translation ±10%, scale 0.85-1.15)
  - Random brightness/contrast shift (thermal intensity scaling)
  - Random Gaussian noise (simulates sensor noise)
  - Random crop (to training resolution)

Reference: UNIP (arXiv:2502.02257) shows grayscale conversion + geometric
augmentation is most effective for IR dense prediction.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def _apply_affine(
    img: np.ndarray,
    mask: np.ndarray | None,
    angle: float,
    translate: tuple[float, float],
    scale: float,
    flip: bool = False,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply affine transform identically to image and mask."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[0, 2] += translate[0] * w
    mat[1, 2] += translate[1] * h

    out_img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
    if flip:
        out_img = cv2.flip(out_img, 1)

    if mask is not None:
        out_mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=border_mode)
        if flip:
            out_mask = cv2.flip(out_mask, 1)
        return out_img, out_mask
    return out_img, None


def _thermal_intensity_shift(img: np.ndarray, brightness: float = 0.1, contrast: float = 0.1) -> np.ndarray:
    """Shift thermal intensity by scale + offset. Image should be float32 [0,1]."""
    if brightness != 0.0:
        img = img + brightness
    if contrast != 0.0:
        mean = img.mean()
        img = (img - mean) * (1.0 + contrast) + mean
    return np.clip(img, 0.0, 1.0)


def _add_gaussian_noise(img: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add thermal sensor noise."""
    if sigma == 0.0:
        return img
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _random_crop(
    img: np.ndarray,
    mask: np.ndarray | None,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random crop to target size."""
    h, w = img.shape[:2]
    if h == crop_h and w == crop_w:
        return img, mask
    if crop_h > h or crop_w > w:
        # Pad if image is smaller
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        h, w = img.shape[:2]

    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    out_img = img[top : top + crop_h, left : left + crop_w]
    if mask is not None:
        out_mask = mask[top : top + crop_h, left : left + crop_w]
        return out_img, out_mask
    return out_img, None


class ThermalAugment:
    """Composable thermal augmentation pipeline.

    Args:
        image_size: Target crop size (square).
        flip_p: Probability of horizontal flip.
        rotation_deg: Max rotation in degrees.
        translate: Max translation as fraction of image size.
        scale_range: Min/max scale factor.
        brightness: Max brightness offset (±).
        contrast: Max contrast scaling factor (±).
        noise_sigma: Max Gaussian noise std dev.
        apply_to_mask: Whether to apply geometric transforms to masks too.
    """

    def __init__(
        self,
        image_size: int = 256,
        flip_p: float = 0.5,
        rotation_deg: float = 15.0,
        translate: float = 0.10,
        scale_range: tuple[float, float] = (0.85, 1.15),
        brightness: float = 0.08,
        contrast: float = 0.10,
        noise_sigma: float = 0.005,
        apply_to_mask: bool = True,
    ):
        self.image_size = image_size
        self.flip_p = flip_p
        self.rotation_deg = rotation_deg
        self.translate = translate
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.noise_sigma = noise_sigma
        self.apply_to_mask = apply_to_mask

    def __call__(self, img: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Args:
            img: [H, W, C] float32 image in [0, 1].
            mask: [H, W] uint8 or float mask (optional).
        Returns:
            (img, mask) augmented, with img resized to image_size.
        """
        # 1. Geometric: affine + flip
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        flip = random.random() < self.flip_p
        img, mask = _apply_affine(img, mask, angle, (tx, ty), scale, flip)

        # 2. Photometric: brightness/contrast
        b = random.uniform(-self.brightness, self.brightness)
        c = random.uniform(-self.contrast, self.contrast)
        img = _thermal_intensity_shift(img, brightness=b, contrast=c)

        # 3. Noise
        sigma = random.uniform(0, self.noise_sigma)
        img = _add_gaussian_noise(img, sigma)

        # 4. Random crop to target size
        img, mask = _random_crop(img, mask, self.image_size, self.image_size)

        return img, mask
