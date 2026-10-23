"""GasVLM: a grounded vision-language model for optical gas imaging.

Pipeline:
    frozen V-JEPA 2 ViT-L (+ low-rank LoRA adapter)
      -> pixel-shuffle projector
      -> SmolLM2-360M-Instruct (+ q/k/v/o LoRA, trainable token embeddings)
      -> multi-scale, [SEG]-conditioned mask decoder (one mask per [SEG])

Auxiliary heads read the language-model hidden state. Two numeric regressors
(CH4%, CO2%) and a [SEG]-emission classifier are supervised only during training
to densify the gradient. The reported concentrations and ratio class come
from the generated answer text, not from these heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gasvlm.models.heads import (
    ScalarRegressor,
    SegEmissionHead,
)
from gasvlm.models.projector import PixelShuffleProjector
from gasvlm.models.special_tokens import NEW_SPECIAL_TOKENS

from .encoder import VJEPA2Encoder


# ---------------------------------------------------------------------------
# Multi-scale mask decoder
# ---------------------------------------------------------------------------

class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + out
        q = q + self.mlp(self.norm2(q))
        return q


class MultiScaleMaskDecoder(nn.Module):
    """Improved mask decoder with multi-scale feature fusion.

    Instead of a single upsample path, we fuse features at multiple scales:
    - Full-resolution visual features (encoder output)
    - Mid-scale from 2×2 strided pooling of visual features
    - Seg hidden state cross-attends at each scale, then fuses.

    This captures both fine-grained edges (gas plume boundaries) and
    coarse spatial structure (source location) more effectively.
    """

    def __init__(
        self,
        seg_dim: int = 960,
        visual_dim: int = 1024,
        inner_dim: int = 256,
        grid_size: int = 16,
        mask_size: int = 256,
        num_layers: int = 2,
        num_scales: int = 2,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.mask_size = mask_size
        self.inner_dim = inner_dim
        self.num_scales = num_scales

        # Project seg hidden to working dim
        self.seg_proj = nn.Linear(seg_dim, inner_dim)

        # Per-scale visual projections
        self.vis_proj_scales = nn.ModuleList()
        self.pos_scales = nn.ParameterList()
        self.cross_blocks = nn.ModuleList()
        self.scale_gates = nn.ModuleList()

        for s in range(num_scales):
            scale_gs = grid_size // (2 ** s)
            vis_in = visual_dim
            self.vis_proj_scales.append(nn.Linear(vis_in, inner_dim))
            self.pos_scales.append(
                nn.Parameter(torch.zeros(1, scale_gs * scale_gs, inner_dim))
            )
            nn.init.trunc_normal_(self.pos_scales[s], std=0.02)
            self.cross_blocks.append(
                nn.ModuleList([_CrossAttnBlock(inner_dim) for _ in range(num_layers)])
            )
            # Learnable gate for scale contribution
            self.scale_gates.append(nn.Sequential(
                nn.Linear(inner_dim, 1),
                nn.Sigmoid(),
            ))

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(inner_dim * num_scales, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # Upsample: grid_size -> mask_size via progressive transposed conv
        # 16 -> 32 -> 64 -> 128 -> 256 (4 stages)
        upsample_layers = []
        in_ch = inner_dim
        n_stages = max(1, int(torch.log2(torch.tensor(mask_size / grid_size)).item()))
        for i in range(n_stages):
            out_ch = max(inner_dim // (2 ** (i + 1)), 32)
            upsample_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            upsample_layers.append(nn.GELU())
            in_ch = out_ch
        upsample_layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))
        self.upsample = nn.Sequential(*upsample_layers)

    def _downsample_visual(self, visual: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample visual tokens by 2^scale via mean pooling on spatial grid."""
        if scale == 0:
            return visual
        b, n, d = visual.shape
        gs = self.grid_size
        # Reshape to [B, gs, gs, D]
        vis_2d = visual.view(b, gs, gs, d)
        # Average pool with stride 2^scale
        kernel = 2 ** scale
        vis_pooled = F.avg_pool2d(
            vis_2d.permute(0, 3, 1, 2), kernel_size=kernel, stride=kernel
        )
        # Flatten back
        return vis_pooled.permute(0, 2, 3, 1).reshape(b, -1, d)

    def forward(
        self, seg_hidden: torch.Tensor, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seg_hidden: [B, seg_dim] — one [SEG] hidden per item.
            visual_tokens: [B, grid_size**2, visual_dim] encoder features.
        Returns:
            mask_logits: [B, 1, mask_size, mask_size]
        """
        b = seg_hidden.shape[0]
        q = self.seg_proj(seg_hidden).unsqueeze(1)  # [B, 1, D]

        scale_feats = []
        for s in range(self.num_scales):
            vis_s = self._downsample_visual(visual_tokens, s)
            n_s = vis_s.shape[1]
            kv = self.vis_proj_scales[s](vis_s) + self.pos_scales[s][:, :n_s, :]
            qs = q
            for blk in self.cross_blocks[s]:
                qs = blk(qs, kv)
            # qs: [B, 1, D]
            # Gate it
            gate = self.scale_gates[s](qs)  # [B, 1, 1]
            scale_feats.append(qs * gate)

        # Concatenate scale features and fuse
        fused = torch.cat(scale_feats, dim=-1)  # [B, 1, D*num_scales]
        fused = self.fusion(fused)  # [B, 1, D]  -- contextualized seg query

        # Gate full-resolution visual features with the seg query, which preserves
        # spatial structure: each position is scaled by the seg query rather than
        # collapsed to a single vector and broadcast uniformly.
        kv_full = self.vis_proj_scales[0](visual_tokens) + self.pos_scales[0]  # [B, N, D]
        gated = kv_full * fused  # [B, N, D] -- broadcast multiply: each spatial position gated by seg query
        feat = gated.transpose(1, 2).reshape(b, self.inner_dim, self.grid_size, self.grid_size)
        x = self.upsample(feat)  # [B, 1, mask_size, mask_size]
        return x


# ---------------------------------------------------------------------------
# Config + Model
# ---------------------------------------------------------------------------

@dataclass
class GasVLMConfig:
    # Encoder
    hf_repo: str = VJEPA2Encoder.DEFAULT_REPO
    img_size: int = 256
    num_image_frames: int = 2
    encoder_freeze: bool = True
    use_lora: bool = True  # single low-rank LoRA adapter on encoder attention projections
    encoder_lora_rank: int = 4
    encoder_lora_alpha: float = 8.0

    # LLM
    llm_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    llm_lora_rank: int = 8
    llm_lora_alpha: float = 16.0
    llm_lora_dropout: float = 0.05
    llm_lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Projector
    pixel_shuffle_r: int = 2

    # Mask decoder
    mask_size: int = 256
    mask_inner_dim: int = 256
    mask_num_layers: int = 2

    # Auxiliary heads
    use_aux_heads: bool = True
    ch4_lo: float = 0.0
    ch4_hi: float = 55.0
    co2_lo: float = 45.0
    co2_hi: float = 100.0

    # Loss weights
    loss_text: float = 1.0
    loss_mask: float = 1.0
    loss_ch4: float = 0.3
    loss_co2: float = 0.3
    loss_emit: float = 0.1


class GasVLM(nn.Module):
    """V-JEPA 2 grounded VLM with a low-rank encoder adapter and a multi-scale mask decoder."""

    def __init__(self, config: GasVLMConfig):
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder(config)

    @staticmethod
    def _build_encoder(config: "GasVLMConfig") -> nn.Module:
        return VJEPA2Encoder(
            hf_repo=config.hf_repo,
            img_size=config.img_size,
            num_image_frames=config.num_image_frames,
            freeze=config.encoder_freeze,
            use_lora=config.use_lora,
            lora_rank=config.encoder_lora_rank,
            lora_alpha=config.encoder_lora_alpha,
        )

    def attach_llm(self, llm: nn.Module, llm_hidden_dim: int) -> None:
        self.llm = llm
        cfg = self.config
        gs = self.encoder.grid_size
        self.projector = PixelShuffleProjector(
            visual_dim=self.encoder.embed_dim,
            llm_dim=llm_hidden_dim,
            grid_size=gs,
            shuffle_r=cfg.pixel_shuffle_r,
        )
        self.mask_decoder = MultiScaleMaskDecoder(
            seg_dim=llm_hidden_dim,
            visual_dim=self.encoder.embed_dim,
            inner_dim=cfg.mask_inner_dim,
            grid_size=gs,
            mask_size=cfg.mask_size,
            num_layers=cfg.mask_num_layers,
            num_scales=2,
        )
        if cfg.use_aux_heads:
            self.ch4_head = ScalarRegressor(
                in_dim=llm_hidden_dim, hidden_dim=256, lo=cfg.ch4_lo, hi=cfg.ch4_hi
            )
            self.co2_head = ScalarRegressor(
                in_dim=llm_hidden_dim, hidden_dim=256, lo=cfg.co2_lo, hi=cfg.co2_hi
            )
            self.emit_head = SegEmissionHead(in_dim=llm_hidden_dim, hidden_dim=256)

    @property
    def has_aux_heads(self) -> bool:
        return self.config.use_aux_heads and self.ch4_head is not None

    def encode_images(self, images: torch.Tensor, camera_type=None) -> torch.Tensor:
        # camera_type is accepted for API symmetry and per-camera eval bookkeeping;
        # the single-LoRA encoder does not condition on it.
        return self.encoder(images)

    def project_visual(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        assert self.projector is not None, "call attach_llm() first"
        return self.projector(visual_tokens)

    def decode_masks(
        self, seg_hidden: torch.Tensor, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        assert self.mask_decoder is not None, "call attach_llm() first"
        return self.mask_decoder(seg_hidden, visual_tokens)

    @torch.no_grad()
    def generate_text(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        camera_type,
        tokenizer,
        max_new_tokens: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.llm is not None, "call attach_llm() first"
        visual = self.encode_images(images, camera_type)
        proj = self.project_visual(visual)

        embed = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([proj, embed], dim=1)
        attn = torch.ones(
            embed.shape[0], proj.shape[1] + prompt_ids.shape[1],
            dtype=torch.long, device=prompt_ids.device,
        )

        eos = getattr(self.llm.config, "eos_token_id", None)
        pad_id = (eos[0] if isinstance(eos, list) else eos) or tokenizer.pad_token_id
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=pad_id,
        )
        return gen_ids, visual, proj

    @torch.no_grad()
    def generate_with_masks(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        camera_type,
        tokenizer,
        seg_token_id: int,
        max_new_tokens: int = 512,
    ):
        assert self.llm is not None, "call attach_llm() first"
        visual = self.encode_images(images, camera_type)
        proj = self.project_visual(visual)

        embed = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([proj, embed], dim=1)
        attn = torch.ones(
            embed.shape[0], proj.shape[1] + prompt_ids.shape[1],
            dtype=torch.long, device=prompt_ids.device,
        )

        eos = getattr(self.llm.config, "eos_token_id", None)
        pad_id = (eos[0] if isinstance(eos, list) else eos) or tokenizer.pad_token_id
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
        )
        gen_ids = out.sequences
        B = visual.shape[0]
        pred_per_item: list = [None] * B
        last_seg_hidden_per_item: list = [None] * B

        if seg_token_id is None or seg_token_id < 0:
            return gen_ids, pred_per_item, last_seg_hidden_per_item

        # Decode each mask from the last-layer state AT its [SEG] token, as in
        # training and the RES eval. generate()'s per-step hidden_states[k]
        # comes from input token k-1, so indexing it by the [SEG] position would
        # read the state of the token before [SEG] ("</p>"). One teacher-forced
        # pass over [visual + prompt + generated tokens] gives the correct states.
        # Assumes no padding inside gen_ids (true for B=1, as the evals run).
        gen_embeds = self.llm.get_input_embeddings()(gen_ids)
        full_embeds = torch.cat([inputs_embeds, gen_embeds], dim=1)
        full_attn = torch.ones(
            full_embeds.shape[:2], dtype=torch.long, device=gen_ids.device,
        )
        fwd = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            output_hidden_states=True,
        )
        last_hidden = fwd.hidden_states[-1]
        n_ctx = inputs_embeds.shape[1]

        for b in range(B):
            seg_pos = (gen_ids[b] == seg_token_id).nonzero(as_tuple=True)[0]
            if len(seg_pos) == 0:
                continue
            seg_h = last_hidden[b, n_ctx + seg_pos]
            vis_for = visual[b].unsqueeze(0).expand(seg_h.shape[0], -1, -1)
            masks = self.decode_masks(seg_h, vis_for)
            pred_per_item[b] = masks
            last_seg_hidden_per_item[b] = seg_h[-1].detach()
        return gen_ids, pred_per_item, last_seg_hidden_per_item
