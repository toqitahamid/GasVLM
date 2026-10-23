"""Frozen V-JEPA 2 ViT-L encoder with a low-rank LoRA adapter for OGI thermal imagery.

A single low-rank adapter on the attention projections of every transformer block
bridges V-JEPA 2's visible-light video pretraining to narrowband MWIR/LWIR thermal
imagery. The base weights stay frozen.

References:
    - HF docs: https://huggingface.co/docs/transformers/model_doc/vjepa2
    - V-JEPA 2 tech report: arXiv 2506.09985
    - LoRA: Hu et al., arXiv 2106.09685
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class _LoRADelta(nn.Module):
    """Single LoRA delta: delta_W = B @ A with rank r."""

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        # B starts at zero so delta is zero at init.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T @ self.B.T) * self.scaling


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a single low-rank LoRA delta.

    The base weight stays frozen; only the low-rank ``A`` / ``B`` factors train.
    ``out = base(x) + scaling * (B @ A) x``.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora = _LoRADelta(in_f, out_f, r=r, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


def _attach_lora_to_vjepa(
    vjepa_model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
) -> list[LoRALinear]:
    """Replace attention linears in the V-JEPA 2 encoder with LoRA wrappers.

    V-JEPA 2 encoder layers use VJEPA2RopeAttention with:
        .query (nn.Linear), .key (nn.Linear), .value (nn.Linear), .proj (nn.Linear)
    We wrap each of these four per layer.
    """
    wrappers: list[LoRALinear] = []
    encoder = getattr(vjepa_model, "encoder", None)
    if encoder is None:
        return wrappers
    layers = getattr(encoder, "layer", None)
    if layers is None:
        return wrappers
    for layer in layers:
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for attr_name in ("query", "key", "value", "proj"):
            lin = getattr(attn, attr_name, None)
            if isinstance(lin, nn.Linear):
                wrap = LoRALinear(lin, r=r, alpha=alpha)
                setattr(attn, attr_name, wrap)
                wrappers.append(wrap)
    return wrappers


class VJEPA2Encoder(nn.Module):
    """Frozen V-JEPA 2 ViT-L with an optional low-rank LoRA adapter for OGI.

    Args:
        hf_repo: HF model id (default ``facebook/vjepa2-vitl-fpc64-256``).
        img_size: Square input resolution.
        num_image_frames: Number of frames in the temporal clip. With a 4-D
            ``[B, 3, H, W]`` input the encoder replicates the single frame
            ``num_image_frames`` times along the time axis (current default
            behavior). With a 5-D ``[B, T, 3, H, W]`` input the encoder skips
            replication and consumes the clip as-is, after asserting
            ``T == num_image_frames``; pair the dataset's
            ``temporal_mode='consecutive_pair'`` with ``num_image_frames=2``
            to feed real ``[prev, curr]`` two-frame clips.
        freeze: Freeze all V-JEPA 2 base parameters (default True).
        use_lora: Attach a low-rank LoRA adapter to the attention q/k/v/proj
            projections of every transformer block.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        dtype: Cast weights to this dtype.
    """

    DEFAULT_REPO = "facebook/vjepa2-vitl-fpc64-256"

    def __init__(
        self,
        hf_repo: str = DEFAULT_REPO,
        img_size: int = 256,
        num_image_frames: int = 2,
        freeze: bool = True,
        use_lora: bool = True,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from transformers import VJEPA2Model

        self.hf_repo = hf_repo
        self.img_size = img_size
        self.num_image_frames = num_image_frames
        self.use_lora = use_lora

        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.vjepa = VJEPA2Model.from_pretrained(hf_repo, **kwargs)

        cfg = self.vjepa.config
        self.embed_dim: int = cfg.hidden_size
        self.patch_size: int = (
            cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
        )
        self.tubelet_size: int = cfg.tubelet_size
        self.crop_size: int = cfg.crop_size

        assert num_image_frames % self.tubelet_size == 0, (
            f"num_image_frames={num_image_frames} must be divisible by tubelet_size={self.tubelet_size}"
        )
        self.temporal_grid: int = num_image_frames // self.tubelet_size
        self.grid_size: int = img_size // self.patch_size

        # ImageNet normalization
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

        # Freeze base
        if freeze:
            for p in self.vjepa.parameters():
                p.requires_grad = False
            self.vjepa.eval()

        # Attach a low-rank LoRA adapter if requested
        self._lora_wrappers: list[LoRALinear] = []
        if use_lora:
            self._lora_wrappers = _attach_lora_to_vjepa(
                self.vjepa, r=lora_rank, alpha=lora_alpha,
            )
            print(f"[VJEPA2Encoder] attached {len(self._lora_wrappers)} LoRA adapters (r={lora_rank}, alpha={lora_alpha})")

    def _prepare_video(self, images: torch.Tensor) -> torch.Tensor:
        """Accept either an image batch or a pre-stacked clip batch.

        * ``[B, 3, H, W]`` (4-D) — single-frame input, replicated
          ``num_image_frames`` times along the time axis (legacy behavior).
        * ``[B, T, 3, H, W]`` (5-D) — pre-stacked clip with ``T`` matching
          ``num_image_frames``, returned as-is after an optional resize.
        """
        if images.ndim == 5:
            b, t, c, h, w = images.shape
            assert c == 3, f"expected 3 channels, got {c}"
            assert t == self.num_image_frames, (
                f"5-D input has T={t} but encoder is configured with "
                f"num_image_frames={self.num_image_frames}; reconfigure the "
                "encoder or the dataset's temporal_mode so they agree."
            )
            if h != self.img_size or w != self.img_size:
                images = F.interpolate(
                    images.reshape(b * t, c, h, w),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, t, c, self.img_size, self.img_size)
            return images.contiguous()

        b, c, h, w = images.shape
        assert c == 3, f"expected 3 channels, got {c}"
        if h != self.img_size or w != self.img_size:
            images = F.interpolate(
                images, size=(self.img_size, self.img_size), mode="bilinear",
                align_corners=False,
            )
        video = images.unsqueeze(1).expand(b, self.num_image_frames, c, self.img_size, self.img_size)
        return video.contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        video = self._prepare_video(images)
        out = self.vjepa(pixel_values_videos=video, skip_predictor=True)
        feats = out.last_hidden_state
        b = feats.shape[0]
        t = self.temporal_grid
        n_spatial = self.grid_size * self.grid_size
        expected = t * n_spatial
        assert feats.shape[1] == expected, (
            f"V-JEPA 2 returned {feats.shape[1]} tokens; expected {expected}"
        )
        if t == 1:
            return feats
        feats = feats.view(b, t, n_spatial, -1).mean(dim=1)
        return feats
