"""Pixel-shuffle visual projector for GasVLM.

Space-to-depth + Linear + LayerNorm, mapping encoder tokens into the language
model's embedding space with a parameter-free shuffle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class PixelShuffleProjector(nn.Module):
    """Pixel-shuffle (r=2) + Linear + LayerNorm.

    Maps ViT patch tokens to LLM tokens via space-to-depth compression.
    SmolVLM finds r=2 preserves spatial topology (needed for mask reconstruction)
    while r=4 hurts localization. Pixel-shuffle is parameter-free.

    Args:
        visual_dim: ViT patch token dim (1024 for V-JEPA 2 ViT-L).
        llm_dim: target LLM hidden dim (960 for SmolLM2-360M).
        grid_size: spatial token grid (16 for ViT-L/16 @ 256).
        shuffle_r: pixel-shuffle ratio (default 2 → 4× channel, /4 tokens).
    """

    def __init__(
        self,
        visual_dim: int = 384,
        llm_dim: int = 960,
        grid_size: int = 24,
        shuffle_r: int = 2,
    ):
        super().__init__()
        assert grid_size % shuffle_r == 0, (
            f"grid_size {grid_size} must be divisible by shuffle_r {shuffle_r}"
        )
        self.grid_size = grid_size
        self.shuffle_r = shuffle_r
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim

        in_dim = visual_dim * shuffle_r * shuffle_r
        self.proj = nn.Linear(in_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D] where N = grid_size**2, D = visual_dim.
        Returns:
            [B, (grid_size/r)**2, llm_dim]
        """
        b, n, d = tokens.shape
        gs, r = self.grid_size, self.shuffle_r
        assert n == gs * gs, f"expected {gs * gs} tokens, got {n}"
        assert d == self.visual_dim, f"expected dim {self.visual_dim}, got {d}"

        # [B, gs, gs, D] -> space-to-depth -> [B, gs/r, gs/r, D*r*r]
        x = tokens.reshape(b, gs, gs, d)
        x = rearrange(
            x, "b (h r1) (w r2) d -> b h w (r1 r2 d)", r1=r, r2=r
        )
        x = x.reshape(b, (gs // r) * (gs // r), d * r * r)
        return self.norm(self.proj(x))
