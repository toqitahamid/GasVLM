"""Lightweight prediction heads driven by [SEG] / response hidden states.

- ScalarRegressor: bounded sigmoid-scaled scalar regressor for CH4% / CO2%.
- SegEmissionHead: per-token binary classifier — "should a [SEG] have been emitted here?"
  Used as an auxiliary loss during training (BCE vs GT positions).

Rationale (primary sources):
    QuantiPhy / MeasureBench / MedVision all show text-token VLM regression
    plateaus due to LM-prior spike values (multiples of 10). Dedicated heads
    bypass tokenizer quantization entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarRegressor(nn.Module):
    """Bounded scalar regressor: sigmoid-scaled to [lo, hi].

    For CH4%: lo=0, hi=55.   For CO2%: lo=45, hi=100.
    """

    def __init__(
        self,
        in_dim: int = 960,
        hidden_dim: int = 256,
        lo: float = 0.0,
        hi: float = 55.0,
    ):
        super().__init__()
        assert hi > lo
        self.lo = lo
        self.hi = hi
        self.head = _MLPHead(in_dim, hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.head(h).squeeze(-1)  # [B]
        return self.lo + (self.hi - self.lo) * torch.sigmoid(x)


class SegEmissionHead(nn.Module):
    """Per-response-token binary classifier for [SEG] emission.

    Input: response-token hidden states [B, T, D].
    Output: logits [B, T] — apply sigmoid for emission probability.
    """

    def __init__(self, in_dim: int = 960, hidden_dim: int = 256):
        super().__init__()
        self.head = _MLPHead(in_dim, hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)  # [B, T]
