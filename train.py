"""GasVLM training loop (DDP via torchrun).

Composite loss:
    L = w_text·CE(LM)
      + w_mask·(BCE+Dice)(masks)
      + w_ch4·Huber(ch4)
      + w_co2·Huber(co2)
      + w_emit·BCE(emit)

Usage:
    torchrun --standalone --nproc_per_node=1 train.py \
        --config configs/gasvlm_headline.yaml \
        [--smoke 500] [--val-loss-every-epoch]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parent

from gasvlm.models.model import GasVLM, GasVLMConfig, NEW_SPECIAL_TOKENS
from gasvlm.data.dataset import build_loader


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(exp_cfg_path: Path) -> dict:
    exp = _load_yaml(exp_cfg_path)
    base_path = exp.get("defaults", {}).get("base")
    if base_path is None:
        return exp
    base = _load_yaml(PROJECT_ROOT / base_path)
    out = base
    _deep_update(out, {k: v for k, v in exp.items() if k != "defaults"})
    return out


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


# ---------------------------------------------------------------------------
# DDP boilerplate
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return rank, world, local, torch.device("cuda", local)
    rank, world, local = 0, 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def seed_everything(seed: int, rank: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices).

    Offset by rank: the dataset draws each sample's record from numpy RNG in the
    DataLoader workers (sample_rates), so equal seeds would give every rank the
    same records. DDP broadcasts rank 0's initial weights, so init is unaffected.
    """
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model / optimizer assembly
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg: dict, device: torch.device, rank: int):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    lcfg = GasVLMConfig(
        hf_repo=mcfg.get("hf_repo", GasVLMConfig.hf_repo),
        img_size=mcfg["img_size"],
        num_image_frames=mcfg.get("num_image_frames", 2),
        encoder_freeze=mcfg.get("encoder_freeze", True),
        use_lora=mcfg.get("use_lora", True),
        encoder_lora_rank=mcfg.get("encoder_lora_rank", 4),
        encoder_lora_alpha=mcfg.get("encoder_lora_alpha", 8.0),
        llm_name=mcfg["llm_name"],
        llm_lora_rank=mcfg["llm_lora_rank"],
        llm_lora_alpha=mcfg["llm_lora_alpha"],
        llm_lora_dropout=mcfg["llm_lora_dropout"],
        llm_lora_targets=tuple(mcfg["llm_lora_targets"]),
        pixel_shuffle_r=mcfg["pixel_shuffle_r"],
        mask_size=mcfg["mask_size"],
        mask_inner_dim=mcfg["mask_inner_dim"],
        mask_num_layers=mcfg["mask_num_layers"],
        use_aux_heads=mcfg.get("use_aux_heads", True),
        ch4_lo=mcfg["ch4_lo"],
        ch4_hi=mcfg["ch4_hi"],
        co2_lo=mcfg["co2_lo"],
        co2_hi=mcfg["co2_hi"],
    )

    log(rank, f"[init] loading tokenizer + LLM: {lcfg.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(lcfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})

    llm = AutoModelForCausalLM.from_pretrained(lcfg.llm_name, dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tokenizer))
    hidden = llm.config.hidden_size

    for p in llm.parameters():
        p.requires_grad = False

    peft_cfg = LoraConfig(
        r=lcfg.llm_lora_rank,
        lora_alpha=lcfg.llm_lora_alpha,
        lora_dropout=lcfg.llm_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lcfg.llm_lora_targets),
    )
    llm = get_peft_model(llm, peft_cfg)

    for name, p in llm.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            p.requires_grad = True

    model = GasVLM(lcfg)
    model.attach_llm(llm, hidden)
    model = model.to(device)
    return model, tokenizer, lcfg


def collect_trainable(model: GasVLM, lr_lora: float, lr_other: float):
    lora_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "llm" in name and ("lora_" in name or "lora_A" in name or "lora_B" in name):
            lora_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora, "name": "llm_lora"})
    if other_params:
        groups.append({"params": other_params, "lr": lr_other, "name": "other"})
    return groups


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


# ---------------------------------------------------------------------------
# Composite loss (same as v1 — compatible with both model classes)
# ---------------------------------------------------------------------------

def composite_step(
    model: GasVLM,
    batch: dict,
    weights: dict,
    seg_token_id: int,
    n_visual_tokens: int,
    device: torch.device,
):
    images = batch["image"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    seg_positions = batch["seg_positions"].to(device, non_blocking=True)
    camera_type = batch["camera_type"]
    ch4_gt = batch["ch4_pct"].to(device, non_blocking=True)
    co2_gt = batch["co2_pct"].to(device, non_blocking=True)
    gt_masks_list = batch["gt_masks"]

    B, L = input_ids.shape

    visual = model.encode_images(images, camera_type)
    proj = model.project_visual(visual)

    embed = model.llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([proj, embed], dim=1)
    visual_attn = torch.ones(B, proj.shape[1], dtype=attention_mask.dtype, device=device)
    full_attn = torch.cat([visual_attn, attention_mask], dim=1)
    visual_labels = torch.full(
        (B, proj.shape[1]), -100, dtype=labels.dtype, device=device
    )
    full_labels = torch.cat([visual_labels, labels], dim=1)

    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attn,
        labels=full_labels,
        output_hidden_states=True,
    )
    L_text = out.loss
    last_hidden = out.hidden_states[-1]
    text_hidden = last_hidden[:, n_visual_tokens:, :]

    seg_b_idx, seg_t_idx = seg_positions.nonzero(as_tuple=True)
    if seg_b_idx.numel() > 0:
        seg_hidden = text_hidden[seg_b_idx, seg_t_idx]
        vis_for_mask = visual[seg_b_idx]
        mask_logits = model.decode_masks(seg_hidden, vis_for_mask)

        per_sample_seg_count: dict[int, int] = {}
        gt_stack = []
        for b, t in zip(seg_b_idx.tolist(), seg_t_idx.tolist()):
            i = per_sample_seg_count.get(b, 0)
            per_sample_seg_count[b] = i + 1
            masks_b = gt_masks_list[b]
            if i < len(masks_b):
                gt_stack.append(masks_b[i])
            else:
                gt_stack.append(torch.zeros(images.shape[-2:], dtype=torch.float32))
        gt_t = torch.stack(gt_stack).to(device)
        if gt_t.shape[-1] != mask_logits.shape[-1]:
            gt_t = F.interpolate(
                gt_t.unsqueeze(1),
                size=mask_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        # Separate BCE/Dice weights if provided, else the combined mask weight.
        pred_logits = mask_logits.squeeze(1) if mask_logits.dim() == 4 and mask_logits.shape[1] == 1 else mask_logits
        target = gt_t.float()
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)
        probs = torch.sigmoid(pred_logits)
        eps = 1e-6
        intersect = (probs * target).flatten(1).sum(-1)
        denom = probs.flatten(1).sum(-1) + target.flatten(1).sum(-1)
        dice = 1.0 - (2.0 * intersect + eps) / (denom + eps)
        # Use separate BCE/Dice weights if provided (LISA recipe), else fall back to combined mask weight
        w_bce = weights.get("mask_bce", weights.get("mask", 1.0))
        w_dice = weights.get("mask_dice", weights.get("mask", 1.0))
        L_mask = w_bce * bce + w_dice * dice.mean()
    else:
        L_mask = torch.zeros((), device=device)
        bce = torch.zeros((), device=device)
        dice = torch.zeros((), device=device)

    if model.has_aux_heads:
        pool_idx = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            positions = seg_positions[b].nonzero(as_tuple=False).flatten()
            if positions.numel() > 0:
                pool_idx[b] = positions[-1]
            else:
                valid = attention_mask[b].nonzero(as_tuple=False).flatten()
                pool_idx[b] = valid[-1] if valid.numel() > 0 else 0
        pooled = text_hidden[torch.arange(B, device=device), pool_idx]
        pooled_fp32 = pooled.float()

        ch4_pred = model.ch4_head(pooled_fp32)
        co2_pred = model.co2_head(pooled_fp32)
        ch4_valid = ch4_gt >= 0
        co2_valid = co2_gt >= 0
        L_ch4 = (
            F.smooth_l1_loss(ch4_pred[ch4_valid], ch4_gt[ch4_valid], beta=2.0)
            if ch4_valid.any() else torch.zeros((), device=device)
        )
        L_co2 = (
            F.smooth_l1_loss(co2_pred[co2_valid], co2_gt[co2_valid], beta=2.0)
            if co2_valid.any() else torch.zeros((), device=device)
        )

        emit_logits = model.emit_head(text_hidden.float())
        response_mask = (labels != -100).float()
        if response_mask.sum() > 0:
            L_emit = F.binary_cross_entropy_with_logits(
                emit_logits, seg_positions.float(),
                weight=response_mask, reduction="sum",
            ) / response_mask.sum()
        else:
            L_emit = torch.zeros((), device=device)

        # Compute total loss
        # L_mask already includes mask_bce and mask_dice weights
        # For backward compat: if weights has "mask" but not "mask_bce"/"mask_dice",
        # we used legacy mode where L_mask was raw BCE+Dice and we multiply by weights["mask"]
        has_separate_mask_weights = "mask_bce" in weights or "mask_dice" in weights
        mask_loss_multiplier = 1.0 if has_separate_mask_weights else weights.get("mask", 1.0)

        L = (
            weights["text"] * L_text
            + mask_loss_multiplier * L_mask
            + weights["ch4"] * L_ch4
            + weights["co2"] * L_co2
            + weights["emit"] * L_emit
        )
    else:
        L_ch4 = L_co2 = L_emit = torch.zeros((), device=device)
        has_separate_mask_weights = "mask_bce" in weights or "mask_dice" in weights
        mask_loss_multiplier = 1.0 if has_separate_mask_weights else weights.get("mask", 1.0)
        L = weights["text"] * L_text + mask_loss_multiplier * L_mask

    return L, {
        "text": L_text.detach().item(),
        "mask": (bce.item() + dice.mean().item()) if isinstance(bce, torch.Tensor) else 0.0,
        "bce": bce.detach().item() if isinstance(bce, torch.Tensor) else 0.0,
        "dice": dice.mean().detach().item() if isinstance(dice, torch.Tensor) else 0.0,
        "ch4": L_ch4.detach().item(),
        "co2": L_co2.detach().item(),
        "emit": L_emit.detach().item(),
        "n_seg": int(seg_b_idx.numel()),
    }


# ---------------------------------------------------------------------------
# Opt-in per-epoch validation-loss pass (training-curve figure)
# ---------------------------------------------------------------------------
#
# Off unless `logging.val_loss_every_epoch: true` or `--val-loss-every-epoch`.
# When off, none of this code runs, so training is unchanged. The val loss is
# only logged; it is never used to pick a checkpoint (final-epoch protocol).
#
# Reduction. Each batch (val and train) goes through the unchanged
# `composite_step`, so every per-batch term uses the training reduction. The
# per-batch mean of each term is turned back into a sum with the count that
# term averages over, then pooled:
#     term = sum_b(mean_b * count_b) / sum_b(count_b)
#   count = response tokens (labels != -100)      for text, emit
#         = [SEG] instances (meters["n_seg"])      for mask, bce, dice
#         = records with a valid label             for ch4, co2
# total  = the training formula applied to the pooled terms.
# mask   = unweighted BCE+Dice (same meter as the step log);
# mask_w = its weighted contribution to total.
# total_batchmean = record-weighted mean of the per-batch loss L (cross-check).
# Val batches are level-pure (one loader per level), so per-level numbers are
# exact.

_VAL_TERMS = ("text", "mask", "bce", "dice", "ch4", "co2", "emit")
_VAL_NFIX = 3  # [records, batches, sum(L * records)]
_VAL_ROW = _VAL_NFIX + 2 * len(_VAL_TERMS)


def _rng_snapshot(device: torch.device) -> dict:
    snap = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        # Current device only: get_rng_state_all() would create a context on
        # every visible GPU.
        snap["cuda"] = torch.cuda.get_rng_state(device)
    return snap


def _rng_restore(snap: dict, device: torch.device) -> None:
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["torch"])
    if "cuda" in snap:
        torch.cuda.set_rng_state(snap["cuda"], device)


def _val_term_counts(batch: dict, meters: dict) -> dict:
    n_tok = int((batch["labels"] != -100).sum().item())
    n_seg = int(meters["n_seg"])
    return {
        "text": n_tok,
        "emit": n_tok,
        "mask": n_seg,
        "bce": n_seg,
        "dice": n_seg,
        "ch4": int((batch["ch4_pct"] >= 0).sum().item()),
        "co2": int((batch["co2_pct"] >= 0).sum().item()),
    }


def _val_acc_row(row: torch.Tensor, loss_item: float, meters: dict, batch: dict) -> None:
    """Add one composite_step result into a float64 accumulator row (CPU)."""
    n_rec = int(batch["input_ids"].shape[0])
    row[0] += n_rec
    row[1] += 1
    row[2] += loss_item * n_rec
    counts = _val_term_counts(batch, meters)
    for ti, k in enumerate(_VAL_TERMS):
        c = counts[k]
        if c > 0:
            row[_VAL_NFIX + 2 * ti] += float(meters[k]) * c
            row[_VAL_NFIX + 2 * ti + 1] += c


def _val_mask_coefs(weights: dict) -> tuple[float, float]:
    """(coef on bce, coef on dice) in the training total, as in composite_step."""
    has_sep = "mask_bce" in weights or "mask_dice" in weights
    mask_mult = 1.0 if has_sep else weights.get("mask", 1.0)
    w_bce = weights.get("mask_bce", weights.get("mask", 1.0))
    w_dice = weights.get("mask_dice", weights.get("mask", 1.0))
    return mask_mult * w_bce, mask_mult * w_dice


def _val_total_from_terms(terms: dict, weights: dict, has_aux: bool) -> dict:
    """Training total (composite_step) rebuilt from pooled terms."""
    t = {k: (v if v is not None else 0.0) for k, v in terms.items()}
    c_bce, c_dice = _val_mask_coefs(weights)
    mask_w = c_bce * t["bce"] + c_dice * t["dice"]
    total = weights["text"] * t["text"] + mask_w
    if has_aux:
        total += (
            weights["ch4"] * t["ch4"] + weights["co2"] * t["co2"]
            + weights["emit"] * t["emit"]
        )
    return {"total": float(total), "mask_w": float(mask_w)}


def _val_summarize(rows: torch.Tensor, weights: dict, has_aux: bool) -> dict:
    s = rows.reshape(-1, _VAL_ROW).sum(dim=0)
    terms = {}
    for ti, k in enumerate(_VAL_TERMS):
        c = float(s[_VAL_NFIX + 2 * ti + 1])
        terms[k] = float(s[_VAL_NFIX + 2 * ti]) / c if c > 0 else None
    n_rec = float(s[0])
    return {
        "records": int(n_rec),
        "batches": int(s[1]),
        **_val_total_from_terms(terms, weights, has_aux),
        "total_batchmean": float(s[2]) / n_rec if n_rec > 0 else None,
        **terms,
        "counts": {k: int(s[_VAL_NFIX + 2 * ti + 1]) for ti, k in enumerate(_VAL_TERMS)},
    }


def _build_val_level_loaders(cfg, tokenizer, rank: int, world: int) -> dict:
    """One sequential, unaugmented, uniformly-indexed loader per VQA level."""
    from functools import partial

    from torch.utils.data import DataLoader, Subset

    from gasvlm.data.dataset import GasVLMDataset, _collate

    ds = GasVLMDataset(
        jsonl_path=cfg["data"]["val_jsonl"],
        tokenizer=tokenizer,
        image_size=cfg["data"]["img_size"],
        max_length=512,
        seg_token="[SEG]",
        # None: __getitem__ honours the index (it is ignored when sample_rates
        # is set), so every val record is seen exactly once.
        sample_rates=None,
        augment=None,
        use_cot=cfg["training"].get("use_cot", False),
        temporal_mode=cfg["data"].get("temporal_mode", "single"),
    )
    by_level: dict[int, list[int]] = {}
    for i, rec in enumerate(ds.records):
        by_level.setdefault(int(rec.get("level", 0)), []).append(i)
    collate = partial(_collate, pad_token_id=int(tokenizer.pad_token_id or 0))
    nw = int(cfg["logging"].get("val_loss_num_workers", cfg["training"]["num_workers"]))
    loaders = {}
    for lv in sorted(by_level):
        idx = by_level[lv][rank::world]  # disjoint strided shard, no padding
        loaders[lv] = DataLoader(
            Subset(ds, idx),
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
            # Dedicated generator: the iterator's worker base_seed draw must
            # not come from the global torch RNG that the train loader uses.
            generator=torch.Generator().manual_seed(0),
        )
    return loaders


def _val_loss_pass(
    m, model, cfg, tokenizer, seg_token_id, n_visual_tokens, device, weights,
    world: int, rank: int, epoch: int, step: int, work_dir: Path, state: dict,
) -> float:
    """Teacher-forced val loss over the full val split. Returns wall seconds.

    Also reports the epoch-mean train loss accumulated in state["train_row"]
    (same pooling), then resets it.
    """
    t_start = time.time()
    if "loaders" not in state:
        state["loaders"] = _build_val_level_loaders(cfg, tokenizer, rank, world)
    loaders = state["loaders"]
    levels = sorted(loaders)

    rng = _rng_snapshot(device)
    modes = [(mod, mod.training) for mod in model.modules()]
    model.eval()

    vec = torch.zeros(len(levels) + 1, _VAL_ROW, dtype=torch.float64)  # last row = train
    try:
        with torch.no_grad():
            for li, lv in enumerate(levels):
                for batch in loaders[lv]:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, meters = composite_step(
                            m, batch, weights, seg_token_id, n_visual_tokens, device
                        )
                    _val_acc_row(vec[li], float(loss.item()), meters, batch)
    finally:
        # Restore each module's own flag. After an epoch every module is in
        # train mode, so this equals model.train(); before training it also
        # keeps the constructor's frozen-encoder eval() until the loop's
        # model.train().
        for mod, was_training in modes:
            mod.train(was_training)
        _rng_restore(rng, device)

    if "train_row" in state:
        vec[-1] = state["train_row"]
        state["train_row"].zero_()

    if world > 1:
        vec_d = vec.to(device)
        dist.all_reduce(vec_d, op=dist.ReduceOp.SUM)
        vec = vec_d.cpu()

    has_aux = bool(m.has_aux_heads)
    out = {"epoch": epoch, "step": step, "world": world}
    out["levels"] = {
        f"L{lv}": _val_summarize(vec[li], weights, has_aux) for li, lv in enumerate(levels)
    }
    out["all"] = _val_summarize(vec[:-1], weights, has_aux)  # val split's own level mix

    rates = cfg["data"].get("sample_rates")
    if rates:
        z = sum(float(v) for v in rates.values())
        w = {int(k): float(v) / z for k, v in rates.items()}
        # (a) levelmix: sum_l w_l * total_l, e.g. 0.4*L1 + 0.3*L2 + 0.3*L3.
        lm = {}
        for k in ("total", "mask_w") + _VAL_TERMS:
            vals = [(w.get(lv, 0.0), out["levels"][f"L{lv}"][k]) for lv in levels]
            lm[k] = (
                sum(wl * v for wl, v in vals if wl > 0 and v is not None)
                / sum(wl for wl, v in vals if wl > 0 and v is not None)
                if any(wl > 0 and v is not None for wl, v in vals) else None
            )
        out["levelmix"] = {"sample_rates": {str(k): v for k, v in w.items()}, **lm}
        # (b) trainmix: pooled as a batch drawn at rates w would pool it
        #     (token/instance-weighted across levels):
        #     term = sum_l w_l*S_l/R_l / sum_l w_l*C_l/R_l   (R_l = records of level l)
        terms = {}
        for ti, k in enumerate(_VAL_TERMS):
            num = den = 0.0
            for li, lv in enumerate(levels):
                r = float(vec[li, 0])
                if r > 0 and w.get(lv, 0.0) > 0:
                    num += w[lv] * float(vec[li, _VAL_NFIX + 2 * ti]) / r
                    den += w[lv] * float(vec[li, _VAL_NFIX + 2 * ti + 1]) / r
            terms[k] = num / den if den > 0 else None
        out["trainmix"] = {
            "sample_rates": {str(k): v for k, v in w.items()},
            **_val_total_from_terms(terms, weights, has_aux),
            **terms,
        }

    if float(vec[-1, 0]) > 0:
        # Epoch-mean of the training loss, train mode, weights changing over
        # the epoch; mixed-level batches drawn at data.sample_rates.
        out["train_epoch"] = _val_summarize(vec[-1], weights, has_aux)

    dt = time.time() - t_start
    out["seconds"] = round(dt, 1)

    if is_main(rank):
        import json

        def _f(x):
            return "nan" if x is None else f"{x:.4f}"

        rows = [(name, out["levels"][name]) for name in out["levels"]]
        rows.append(("all", out["all"]))
        for key in ("levelmix", "trainmix", "train_epoch"):
            if key in out:
                rows.append((key, out[key]))
        for name, r in rows:
            log(
                rank,
                f"[val] ep={epoch} step={step} {name} L={_f(r['total'])} "
                f"Lbm={_f(r.get('total_batchmean'))} "
                f"text={_f(r['text'])} mask={_f(r['mask'])} mask_w={_f(r['mask_w'])} "
                f"ch4={_f(r['ch4'])} co2={_f(r['co2'])} emit={_f(r['emit'])}"
                + (f" n={r['records']}" if "records" in r else ""),
            )
        log(rank, f"[val] ep={epoch} pass took {dt:.0f}s")
        path = work_dir / "val_curve.jsonl"
        mode = "a" if state.get("wrote") else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        state["wrote"] = True

    if world > 1:
        dist.barrier()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, run only this many optimizer steps and exit")
    ap.add_argument("--val-loss-every-epoch", action="store_true",
                    help="opt-in: teacher-forced loss over the full val split before training "
                         "and after every epoch -> work_dir/val_curve.jsonl (same as "
                         "logging.val_loss_every_epoch: true)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    rank, world, local_rank, device = setup_ddp()

    # Unset (the default) leaves torch on its built-in default seed, as before.
    seed = cfg["training"].get("seed")
    if seed is not None:
        seed_everything(int(seed), rank)

    work_dir = Path(cfg["logging"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    log(rank, f"[init] world_size={world} rank={rank} device={device}")
    log(rank, f"[cfg] work_dir={work_dir}")
    if seed is not None:
        log(rank, f"[init] seed={int(seed)} (+rank offset)")

    # Build model
    model, tokenizer, _ = build_model_and_tokenizer(cfg, device, rank)
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    # Derive from the actual encoder so non-vjepa2 encoders (e.g. I-JEPA patch=14)
    # don't silently fall back to the hardcoded `// 16` patch-size assumption.
    n_visual_tokens = (
        model.encoder.grid_size // cfg["model"]["pixel_shuffle_r"]
    ) ** 2

    # Param counting
    if is_main(rank):
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(rank, f"[init] params total={n_total / 1e6:.1f}M trainable={n_train / 1e6:.1f}M")

    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        m = model.module
    else:
        m = model

    # Optimizer
    groups = collect_trainable(m, cfg["training"]["lr_lora"], cfg["training"]["lr_other"])
    optim = torch.optim.AdamW(
        groups, weight_decay=cfg["training"]["weight_decay"], betas=(0.9, 0.95)
    )

    # Data
    augment = None
    if cfg["training"].get("use_thermal_augment", False):
        from gasvlm.data.thermal_augment import ThermalAugment
        augment = ThermalAugment(image_size=cfg["data"]["img_size"])
        if is_main(rank):
            log(rank, "[data] thermal augmentation enabled")

    use_cot = cfg["training"].get("use_cot", False)
    train_loader, train_sampler = build_loader(
        jsonl_path=cfg["data"]["train_jsonl"],
        tokenizer=tokenizer,
        image_size=cfg["data"]["img_size"],
        max_length=512,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        sample_rates=cfg["data"]["sample_rates"],
        shuffle=True,
        distributed=(world > 1),
        augment=augment,
        use_cot=use_cot,
        temporal_mode=cfg["data"].get("temporal_mode", "single"),
    )
    log(rank, f"[data] train batches/epoch = {len(train_loader)}")

    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["grad_accum"]
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    if args.smoke:
        total_steps = args.smoke
    warmup = cfg["training"]["warmup_steps"]
    weights = cfg["losses"]

    # Opt-in per-epoch val-loss pass (default off; off => nothing below runs).
    val_loss_on = bool(
        args.val_loss_every_epoch or cfg["logging"].get("val_loss_every_epoch", False)
    )
    if val_loss_on and args.smoke:
        log(rank, "[val] val_loss_every_epoch ignored in --smoke mode")
        val_loss_on = False
    val_state: dict = {}
    if val_loss_on:
        val_state["train_row"] = torch.zeros(_VAL_ROW, dtype=torch.float64)
    if val_loss_on and cfg["logging"].get("val_loss_epoch0", True):
        _val_loss_pass(
            m, model, cfg, tokenizer, seg_token_id, n_visual_tokens, device, weights,
            world, rank, 0, 0, work_dir, val_state,
        )

    # Train loop
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for it, batch in enumerate(train_loader):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, meters = composite_step(
                    m, batch, weights, seg_token_id, n_visual_tokens, device
                )
            if val_loss_on:  # epoch-mean train loss for the matching train curve
                _val_acc_row(val_state["train_row"], float(loss.item()), meters, batch)
            (loss / grad_accum).backward()

            if (it + 1) % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], cfg["training"]["grad_clip"]
            )
            scale = cosine_lr(step, total_steps, warmup)
            for g in groups:
                base_lr = (
                    cfg["training"]["lr_lora"]
                    if g["name"] == "llm_lora"
                    else cfg["training"]["lr_other"]
                )
                g["lr"] = base_lr * scale
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if is_main(rank) and step % cfg["logging"]["log_every_n_steps"] == 0:
                lr_now = optim.param_groups[0]["lr"]
                el = time.time() - t0
                log(
                    rank,
                    f"[step {step}/{total_steps}] ep={epoch} lr={lr_now:.2e} "
                    f"L={loss.item():.3f} text={meters['text']:.3f} "
                    f"mask={meters['mask']:.3f} "
                    f"ch4={meters['ch4']:.3f} co2={meters['co2']:.3f} "
                    f"emit={meters['emit']:.3f} nseg={meters['n_seg']} "
                    f"({el:.0f}s)",
                )

            if args.smoke and step >= args.smoke:
                log(rank, f"[smoke] reached step {step}, exiting")
                if world > 1:
                    dist.barrier()
                return

        # Checkpoint: save FULL state dict (frozen encoder weights needed at eval)
        if is_main(rank):
            ckpt_dir = work_dir / f"epoch_{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sd = m.state_dict()
            torch.save(sd, ckpt_dir / "gasvlm.pt")
            log(rank, f"[ckpt] saved {ckpt_dir / 'gasvlm.pt'} ({len(sd)} keys, full state dict)")

        if val_loss_on:
            val_dt = _val_loss_pass(
                m, model, cfg, tokenizer, seg_token_id, n_visual_tokens, device, weights,
                world, rank, epoch + 1, step, work_dir, val_state,
            )
            t0 += val_dt  # keep the "(Ns)" in step logs = training time only

    log(rank, "[done] training complete")


if __name__ == "__main__":
    main()
