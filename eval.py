"""GasVLM per-level evaluation: RES (L1 segmentation), VQA (L2), GCG (L3).

Reconstructs the model from the YAML config, loads a checkpoint, and reports
the per-level metrics under distributed data parallelism. Run under torchrun
(see dist_eval.sh), even for a single GPU.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent

import yaml  # noqa: E402

from gasvlm.data.vqa_dataset import VQADataset, _extract_binary_masks  # noqa: E402
from gasvlm.models.model import GasVLM, GasVLMConfig, NEW_SPECIAL_TOKENS  # noqa: E402

from gasvlm.utils.dist import (  # noqa: E402
    _init_dist_pytorch,
    collect_results_cpu,
    get_dist_info,
)
from gasvlm.utils.eval_metrics import (  # noqa: E402
    _compute_gcg_captioning_metrics,
    _compute_gcg_mask_ap,
    _compute_gcg_recall,
    _compute_miou_one_to_one,
    _parse_concentrations,
    _parse_ratio_class,
    _ratio_class_metrics,
    _regression_metrics,
)


# --- Config / IO ----------------------------------------------------------------

# Default decode path is the captured one; --no-fast-decode restores eager.
FAST_DECODE_OFF = False


def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
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


def load_jsonl(jsonl_path: str, level=None, tasks=None) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            if level is not None and it.get("level") != level:
                continue
            if tasks is not None and it.get("task") not in tasks:
                continue
            out.append(it)
    return out


def subsample(items, max_samples: int):
    """Take max_samples entries spread evenly across `items`.

    The split files are ordered by video, so items[:N] draws from only the first
    one or two gas releases. Striding spreads the subset over the whole split
    (all 14 test videos from about 500 records per level) and is deterministic
    with no RNG.
    """
    if max_samples <= 0 or len(items) <= max_samples:
        return items
    return items[:: len(items) // max_samples][:max_samples]


# --- Mask RLE helpers (for L3 cross-rank gather) --------------------------------

def _mask_to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _rle_to_mask(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils
    r = {
        "size": rle["size"],
        "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"],
    }
    return mask_utils.decode(r).astype(np.uint8)


# --- Model loading --------------------------------------------------------------

def build_model(cfg: dict, ckpt_path: str, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    lcfg = GasVLMConfig(
        hf_repo=mcfg.get("hf_repo", GasVLMConfig.hf_repo),
        img_size=mcfg["img_size"],
        num_image_frames=mcfg.get("num_image_frames", 2),
        encoder_freeze=True,
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

    tokenizer = AutoTokenizer.from_pretrained(lcfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})

    llm = AutoModelForCausalLM.from_pretrained(lcfg.llm_name, dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tokenizer))
    hidden = llm.config.hidden_size

    peft_cfg = LoraConfig(
        r=lcfg.llm_lora_rank,
        lora_alpha=lcfg.llm_lora_alpha,
        lora_dropout=lcfg.llm_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lcfg.llm_lora_targets),
    )
    llm = get_peft_model(llm, peft_cfg)

    model = GasVLM(lcfg)
    model.attach_llm(llm, hidden)
    model = model.to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "vlm_state_dict")):
        for k in ("vlm_state_dict", "state_dict", "model"):
            if k in sd:
                sd = sd[k]
                break
    # strict=True: a partial or wrong checkpoint must fail loudly rather than
    # evaluate a half-initialized model and report plausible-looking numbers.
    model.load_state_dict(sd, strict=True)

    model.eval()
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    return model, tokenizer, seg_token_id, lcfg


# --- Forward helpers ------------------------------------------------------------

@torch.no_grad()
def _forward_with_teacher_forced_seg(
    model: GasVLM, batch: dict, seg_token_id: int, device: torch.device,
):
    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    cam = batch["camera_type"]
    seg_pos = batch["seg_positions"].to(device)

    visual = model.encode_images(images, cam)
    proj = model.project_visual(visual)
    n_v = proj.shape[1]

    embed = model.llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([proj, embed], dim=1)
    visual_attn = torch.ones(embed.shape[0], n_v, dtype=attn.dtype, device=device)
    full_attn = torch.cat([visual_attn, attn], dim=1)
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attn,
        output_hidden_states=True,
    )
    last_hidden = out.hidden_states[-1]
    text_hidden = last_hidden[:, n_v:, :]

    seg_b_idx, seg_t_idx = seg_pos.nonzero(as_tuple=True)
    if seg_b_idx.numel() > 0:
        seg_hidden = text_hidden[seg_b_idx, seg_t_idx]
        vis_for_mask = visual[seg_b_idx]
        mask_logits = model.decode_masks(seg_hidden, vis_for_mask)
    else:
        mask_logits = None

    return {
        "text_hidden": text_hidden,
        "visual": visual,
        "seg_b_idx": seg_b_idx,
        "seg_t_idx": seg_t_idx,
        "mask_logits": mask_logits,
        "n_visual_tokens": n_v,
        "input_ids": input_ids,
        "attn": attn,
        "proj": proj,
    }


@torch.no_grad()
def _generate_text(
    model: GasVLM, sample: dict, tokenizer, device: torch.device,
    max_new_tokens: int = 128, max_prompt_len: int = 384,
):
    images = sample["image"].unsqueeze(0).to(device)
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    labels = sample["labels"].unsqueeze(0).to(device)
    cam = [sample["camera_type"]]

    response_start = (labels[0] != -100).nonzero(as_tuple=False)
    if response_start.numel() > 0:
        prompt_len = int(response_start[0].item())
    else:
        prompt_len = input_ids.shape[1]
    prompt_ids = input_ids[:, :prompt_len][:, :max_prompt_len]

    visual = model.encode_images(images, cam)
    proj = model.project_visual(visual)
    n_v = proj.shape[1]

    embed = model.llm.get_input_embeddings()(prompt_ids)
    inputs_embeds = torch.cat([proj, embed], dim=1)
    attn = torch.ones(1, n_v + prompt_ids.shape[1], dtype=torch.long, device=device)

    gen = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = gen[0]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return text, visual, proj


# --- Level 1 inference + metrics ----------------------------------------------

def _run_level1(model, tokenizer, seg_token_id, full_ds, indices, device,
                rank: int, world_size: int):
    n = len(indices)
    per_rank = math.ceil(n / world_size)
    my = indices[per_rank * rank: per_rank * (rank + 1)]

    K = 3
    task_to_class = {"seg_source": 1, "seg_gas": 2}
    records = []

    for idx in tqdm(my, desc=f"L1 rank{rank}", disable=(rank != 0)):
        sample = full_ds[idx]
        rec = full_ds.records[idx]
        task = rec["task"]
        cls_idx = task_to_class[task]

        from torch.utils.data._utils.collate import default_collate
        batch = {
            "image": sample["image"].unsqueeze(0),
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "labels": sample["labels"].unsqueeze(0),
            "seg_positions": sample["seg_positions"].unsqueeze(0),
            "camera_type": [rec.get("camera_type", "co2")],
        }
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = _forward_with_teacher_forced_seg(model, batch, seg_token_id, device)

        mask_path = rec.get("mask_path", "")
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            mask_raw = cv2.imread(str(PROJECT_ROOT / mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            continue
        gt_h, gt_w = mask_raw.shape
        if task == "seg_gas":
            gt_bin = (mask_raw == 255).astype(np.uint8)
        else:
            gt_bin = (mask_raw == 100).astype(np.uint8)

        gt_kclass = np.zeros_like(mask_raw, dtype=np.int64)
        gt_kclass[gt_bin.astype(bool)] = cls_idx

        if out["mask_logits"] is not None and out["mask_logits"].shape[0] > 0:
            pred_logits = out["mask_logits"][0, 0]
            pred_resized = F.interpolate(
                pred_logits.unsqueeze(0).unsqueeze(0).float(),
                size=(gt_h, gt_w), mode="bilinear", align_corners=False,
            ).squeeze()
            pred_bin = (pred_resized > 0).long()
            pred_kclass = torch.zeros(gt_h, gt_w, dtype=torch.long, device=device)
            pred_kclass[pred_bin.bool()] = cls_idx
        else:
            pred_kclass = torch.zeros(gt_h, gt_w, dtype=torch.long, device=device)
            pred_bin = torch.zeros(gt_h, gt_w, dtype=torch.long, device=device)

        gt_kclass_t = torch.from_numpy(gt_kclass).long().to(device)
        i_arr = np.zeros(K, dtype=np.float64)
        u_arr = np.zeros(K, dtype=np.float64)
        for c in range(K):
            pred_c = (pred_kclass == c)
            gt_c = (gt_kclass_t == c)
            i_arr[c] = (pred_c & gt_c).sum().item()
            u_arr[c] = (pred_c | gt_c).sum().item()
        cls_u = u_arr[cls_idx]
        sample_iou = float(i_arr[cls_idx] / (cls_u + 1e-10)) if cls_u > 0 else 1.0

        rc_val = rec.get("ratio_class_5")
        rc = str(rc_val) if rc_val is not None else "unknown"
        cam = rec.get("camera_type", "unknown")
        gt_has = bool(int(gt_bin.sum()) > 0) if task == "seg_gas" else False
        pred_has = bool(pred_bin.sum().item() > 0) if task == "seg_gas" else False

        records.append({
            "task": task,
            "class_idx": cls_idx,
            "intersection": i_arr.tolist(),
            "union": u_arr.tolist(),
            "sample_iou": sample_iou,
            "ratio_class": rc,
            "camera_type": cam,
            "gt_has_gas": gt_has,
            "pred_has_gas": pred_has,
        })

    return records


def _metrics_level1(records, num_total: int):
    K = 3
    inter = np.zeros(K, dtype=np.float64)
    union = np.zeros(K, dtype=np.float64)
    giou_sum = 0.0
    giou_count = 0
    by_ratio = defaultdict(lambda: {"inter": 0.0, "union": 0.0, "iou_sum": 0.0, "n": 0})
    by_camera = defaultdict(lambda: {"inter": 0.0, "union": 0.0, "iou_sum": 0.0, "n": 0})
    gas_tp = gas_fp = gas_tn = gas_fn = 0

    for r in records:
        i = np.array(r["intersection"]); u = np.array(r["union"])
        inter += i; union += u
        giou_sum += r["sample_iou"]; giou_count += 1
        ci = r["class_idx"]
        rc = r["ratio_class"]; cam = r["camera_type"]
        by_ratio[rc]["inter"] += i[ci]; by_ratio[rc]["union"] += u[ci]
        by_ratio[rc]["iou_sum"] += r["sample_iou"]; by_ratio[rc]["n"] += 1
        by_camera[cam]["inter"] += i[ci]; by_camera[cam]["union"] += u[ci]
        by_camera[cam]["iou_sum"] += r["sample_iou"]; by_camera[cam]["n"] += 1
        if r["task"] == "seg_gas":
            gh, ph = r["gt_has_gas"], r["pred_has_gas"]
            if gh and ph: gas_tp += 1
            elif (not gh) and ph: gas_fp += 1
            elif (not gh) and (not ph): gas_tn += 1
            else: gas_fn += 1

    cIoU_per = (inter / np.clip(union, 1e-10, None)).tolist()
    mIoU_3 = float(np.mean(cIoU_per))
    overall_inter = inter[1:].sum() + inter[0]
    overall_union = union[1:].sum() + union[0]
    cIoU_overall = float((inter[1] + inter[2]) / max(1e-10, (union[1] + union[2])))
    gIoU = float(giou_sum / max(1, giou_count))

    dice_per = [float(2 * inter[c] / max(1e-10, inter[c] + union[c])) for c in range(K)]
    mean_dice = float(np.mean(dice_per[1:]))

    by_ratio_out = {}
    for rc, d in by_ratio.items():
        by_ratio_out[rc] = {
            "n": d["n"],
            "cIoU": float(d["inter"] / max(1e-10, d["union"])),
            "gIoU": float(d["iou_sum"] / max(1, d["n"])),
        }
    by_camera_out = {}
    for cam, d in by_camera.items():
        by_camera_out[cam] = {
            "n": d["n"],
            "cIoU": float(d["inter"] / max(1e-10, d["union"])),
            "gIoU": float(d["iou_sum"] / max(1, d["n"])),
        }

    gas_total = gas_tp + gas_fp + gas_tn + gas_fn
    gas_acc = float((gas_tp + gas_tn) / max(1, gas_total))

    return {
        "num_samples": num_total,
        "num_records_evaluated": len(records),
        "overall": {
            "gIoU": gIoU, "cIoU": cIoU_overall, "mIoU_3class": mIoU_3,
            "mean_Dice": mean_dice,
        },
        "per_class": {
            "background": {"IoU": cIoU_per[0], "Dice": dice_per[0]},
            "source": {"IoU": cIoU_per[1], "Dice": dice_per[1]},
            "gas": {"IoU": cIoU_per[2], "Dice": dice_per[2]},
        },
        "per_ratio_class": by_ratio_out,
        "per_camera": by_camera_out,
        "gas_detection": {
            "tp": gas_tp, "fp": gas_fp, "tn": gas_tn, "fn": gas_fn,
            "accuracy": gas_acc,
        },
    }


def eval_level1(model, tokenizer, seg_token_id, cfg, device, test_jsonl,
                max_samples, rank, world_size):
    if rank == 0:
        print("\n" + "=" * 60); print("LEVEL 1: SEGMENTATION"); print("=" * 60)
    full_ds = VQADataset(
        jsonl_path=test_jsonl, tokenizer=tokenizer,
        image_size=cfg["data"]["img_size"], max_length=512, seg_token="[SEG]",
        temporal_mode=cfg["data"].get("temporal_mode", "single"),
    )
    indices = [i for i, r in enumerate(full_ds.records)
               if r.get("level") == 1 and r.get("task") in {"seg_gas", "seg_source"}]
    if max_samples > 0:
        indices = subsample(indices, max_samples)
    if rank == 0:
        print(f"Evaluating {len(indices)} L1 samples across {world_size} rank(s)")

    records = _run_level1(model, tokenizer, seg_token_id, full_ds, indices,
                          device, rank, world_size)

    if world_size > 1:
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or "local"
        tmpdir = f"/tmp/gasvlm_l1_gather_{job_id}"
        records = collect_results_cpu(records, len(indices), tmpdir=tmpdir)
        if rank != 0:
            return None
    return _metrics_level1(records, num_total=len(indices))


# --- Level 2 inference + metrics ----------------------------------------------

def _run_level2(model, tokenizer, seg_token_id, full_ds, indices, device,
                rank: int, world_size: int):
    n = len(indices)
    per_rank = math.ceil(n / world_size)
    my = indices[per_rank * rank: per_rank * (rank + 1)]
    records = []

    for idx in tqdm(my, desc=f"L2 rank{rank}", disable=(rank != 0)):
        sample = full_ds[idx]
        rec = full_ds.records[idx]
        task = rec["task"]
        cam = rec.get("camera_type", "co2")

        q_len = int(sample["q_len"])
        prompt_ids = sample["input_ids"][:q_len].unsqueeze(0).to(device)
        images = sample["image"].unsqueeze(0).to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            gen_ids, _, _ = model.generate_text(
                images, prompt_ids, [cam], tokenizer, max_new_tokens=128,
            )

        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        gen_ratio = _parse_ratio_class(gen_text)
        gen_ch4, gen_co2 = _parse_concentrations(gen_text)

        records.append({
            "task": task,
            "camera_type": cam,
            "ratio_class_gt": rec.get("ratio_class_5"),
            "ch4_gt": rec.get("ch4_pct"),
            "co2_gt": rec.get("co2_pct"),
            "answer_gt": rec.get("answer", ""),
            "gen_text": gen_text,
            "gen_ratio": gen_ratio if gen_ratio is not None else -1,
            "gen_ch4": gen_ch4,
            "gen_co2": gen_co2,
        })

    return records


def _metrics_level2(records, num_total):
    out = {"num_samples": num_total, "num_records_evaluated": len(records)}

    rc_recs = [r for r in records if r["task"] == "ratio_class" and r["ratio_class_gt"] is not None]
    co_recs = [r for r in records if r["task"] == "concentration"
               and r["ch4_gt"] is not None and r["co2_gt"] is not None]

    if rc_recs:
        gt = [int(r["ratio_class_gt"]) for r in rc_recs]

        gen_pr_raw = [int(r["gen_ratio"]) for r in rc_recs]
        parse_fails = sum(1 for p in gen_pr_raw if p < 1)
        gen_pr = [p if p >= 1 else 0 for p in gen_pr_raw]
        gen_m = _ratio_class_metrics(gt, gen_pr)
        gen_m["total"] = len(rc_recs)
        gen_m["parse_fail"] = parse_fails
        gen_m["parse_fail_rate"] = parse_fails / max(1, len(rc_recs))
        gen_m["within_1_accuracy"] = float(
            np.mean([abs(g - p) <= 1 for g, p in zip(gt, gen_pr) if p >= 1] or [0.0])
        )

        rc_out = {"gen": gen_m, "total": len(rc_recs)}
        rc_out["accuracy"] = gen_m["accuracy"]
        rc_out["within_1_accuracy"] = gen_m["within_1_accuracy"]
        rc_out["macro_f1"] = gen_m["macro_f1"]
        out["ratio_class"] = rc_out

    if co_recs:
        gen_pairs_ch4 = [(r["gen_ch4"], r["ch4_gt"]) for r in co_recs if r["gen_ch4"] is not None]
        gen_pairs_co2 = [(r["gen_co2"], r["co2_gt"]) for r in co_recs if r["gen_co2"] is not None]
        gen_ch4_m = (_regression_metrics([float(p) for p, _ in gen_pairs_ch4],
                                         [float(g) for _, g in gen_pairs_ch4])
                     if gen_pairs_ch4 else {})
        gen_co2_m = (_regression_metrics([float(p) for p, _ in gen_pairs_co2],
                                         [float(g) for _, g in gen_pairs_co2])
                     if gen_pairs_co2 else {})

        co_out = {
            "gen": {
                "ch4": gen_ch4_m, "co2": gen_co2_m,
                "ch4_mae": gen_ch4_m.get("mae"), "ch4_mape": gen_ch4_m.get("mape"),
                "co2_mae": gen_co2_m.get("mae"), "co2_mape": gen_co2_m.get("mape"),
                "ch4_parse_fail": len(co_recs) - len(gen_pairs_ch4),
                "co2_parse_fail": len(co_recs) - len(gen_pairs_co2),
                "n": len(co_recs),
            },
            "n": len(co_recs),
        }

        co_out["ch4_mae"] = gen_ch4_m.get("mae")
        co_out["ch4_mape"] = gen_ch4_m.get("mape")
        co_out["co2_mae"] = gen_co2_m.get("mae")
        co_out["co2_mape"] = gen_co2_m.get("mape")

        out["concentration"] = co_out

    gt_caps: dict[int, str] = {}
    pred_caps: dict[int, str] = {}
    img_ids: list[int] = []
    for i, r in enumerate(records):
        gt = (r.get("answer_gt") or "").strip()
        if not gt:
            continue
        img_ids.append(i)
        gt_caps[i] = gt
        pred_caps[i] = (r.get("gen_text") or "").strip()
    if img_ids:
        try:
            out["captioning"] = _compute_gcg_captioning_metrics(
                gt_caps, pred_caps, img_ids,
            )
        except Exception as e:
            out["captioning_error"] = str(e)

    return out


def eval_level2(model, tokenizer, seg_token_id, cfg, device, test_jsonl,
                max_samples, rank, world_size):
    if rank == 0:
        print("\n" + "=" * 60); print("LEVEL 2: VQA"); print("=" * 60)
    full_ds = VQADataset(
        jsonl_path=test_jsonl, tokenizer=tokenizer,
        image_size=cfg["data"]["img_size"], max_length=512, seg_token="[SEG]",
        temporal_mode=cfg["data"].get("temporal_mode", "single"),
    )
    indices = [i for i, r in enumerate(full_ds.records) if r.get("level") == 2]
    if max_samples > 0:
        indices = subsample(indices, max_samples)
    if rank == 0:
        print(f"Evaluating {len(indices)} L2 samples across {world_size} rank(s)")

    records = _run_level2(model, tokenizer, seg_token_id, full_ds, indices,
                          device, rank, world_size)
    if world_size > 1:
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or "local"
        tmpdir = f"/tmp/gasvlm_l2_gather_{job_id}"
        records = collect_results_cpu(records, len(indices), tmpdir=tmpdir)
        if rank != 0:
            return None
    return _metrics_level2(records, num_total=len(indices))


# --- Level 3 inference + metrics ----------------------------------------------

_PHRASE_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def _parse_gcg_phrases(text: str) -> list[str]:
    return [p.strip().lower() for p in _PHRASE_RE.findall(text)]


def _run_level3(model, tokenizer, seg_token_id, full_ds, indices, device,
                rank, world_size):
    n = len(indices)
    per_rank = math.ceil(n / world_size)
    my = indices[per_rank * rank: per_rank * (rank + 1)]
    records = []

    for global_idx in tqdm(my, desc=f"L3 rank{rank}", disable=(rank != 0)):
        sample = full_ds[global_idx]
        rec = full_ds.records[global_idx]
        task = rec["task"]
        cam = rec.get("camera_type", "co2")
        mask_classes = rec.get("mask_classes") or []

        q_len = int(sample["q_len"])
        prompt_ids = sample["input_ids"][:q_len].unsqueeze(0).to(device)
        images = sample["image"].unsqueeze(0).to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            gen_ids, pred_per_item, last_seg_hidden_per_item = model.generate_with_masks(
                images, prompt_ids, [cam], tokenizer, seg_token_id,
                max_new_tokens=512,
            )

        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False).strip()
        cleaned = re.sub(r"<.*?>", "", gen_text).replace("[SEG]", "").strip()
        cleaned = " ".join(cleaned.split())

        pred_phrases = _parse_gcg_phrases(gen_text)
        pred_ratio = _parse_ratio_class(gen_text)
        pred_ch4, pred_co2 = _parse_concentrations(gen_text)

        pred_masks_t = pred_per_item[0]
        pred_masks_np = []
        if pred_masks_t is not None:
            for k in range(pred_masks_t.shape[0]):
                pred_masks_np.append(
                    (pred_masks_t[k, 0] > 0).cpu().numpy().astype(np.uint8)
                )

        mask_path = rec.get("mask_path", "")
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            mask_raw = cv2.imread(str(PROJECT_ROOT / mask_path), cv2.IMREAD_GRAYSCALE)
        gt_masks_np = []
        gt_phrases = []
        if mask_raw is not None and mask_classes:
            gt_bins = _extract_binary_masks(mask_raw, mask_classes)
            gt_masks_np = [gb.astype(np.uint8) for gb in gt_bins]
            gt_phrases = [mc.lower() for mc in mask_classes]

        pred_masks_for_metric = pred_masks_np[:len(pred_phrases)] if pred_phrases else pred_masks_np

        preds_resized = []
        if pred_masks_for_metric and gt_masks_np:
            gt_h, gt_w = gt_masks_np[0].shape
            for pm in pred_masks_for_metric:
                if pm.shape != (gt_h, gt_w):
                    pm_r = cv2.resize(pm, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                else:
                    pm_r = pm
                preds_resized.append(pm_r)
            per_img_miou = _compute_miou_one_to_one(preds_resized, gt_masks_np)
        else:
            per_img_miou = 0.0

        gt_rle = [_mask_to_rle(m) for m in gt_masks_np]
        pred_rle = [_mask_to_rle(m) for m in (preds_resized if preds_resized else pred_masks_for_metric)]
        gt_gas_rle = (_mask_to_rle((mask_raw == 255).astype(np.uint8))
                      if mask_raw is not None else None)
        gt_gas_present = bool((mask_raw == 255).any()) if mask_raw is not None else False

        records.append({
            "image_id": global_idx,
            "task": task,
            "camera_type": cam,
            "ratio_class_5": rec.get("ratio_class_5"),
            "ch4_pct": rec.get("ch4_pct"),
            "co2_pct": rec.get("co2_pct"),
            "gen_text": gen_text,
            "cleaned_caption": cleaned,
            "gt_caption": rec.get("answer_clean", rec.get("answer", "")),
            "pred_phrases": pred_phrases,
            "gt_phrases": gt_phrases,
            "pred_ratio": pred_ratio if pred_ratio is not None else -1,
            "pred_ch4": pred_ch4, "pred_co2": pred_co2,
            "n_seg_gt": len(mask_classes),
            "n_seg_pred": len(pred_masks_np),
            "seg_token_found": bool(pred_masks_t is not None and pred_masks_t.shape[0] > 0),
            "per_image_miou": per_img_miou,
            "gt_masks_rle": gt_rle,
            "pred_masks_rle": pred_rle,
            "gt_gas_rle": gt_gas_rle,
            "gt_gas_present": gt_gas_present,
        })

    return records


def _metrics_level3(records, num_total):
    by_task = defaultdict(list)
    by_camera = defaultdict(list)
    all_miou = []
    seg_emit_count = 0
    n_seg_gt_total = 0
    n_seg_pred_total = 0

    gt_masks_per_image = {}
    pred_masks_per_image = {}
    gt_phrases_per_image = {}
    pred_phrases_per_image = {}
    gt_captions = {}
    pred_captions = {}

    rc_gt, rc_pred = [], []
    rc_parse_fails = 0
    ch4_pairs, co2_pairs = [], []
    ch4_parse_fails = co2_parse_fails = 0

    gas_det_tp = gas_det_fp = gas_det_tn = gas_det_fn = 0

    for r in records:
        img_id = r["image_id"]
        n_seg_gt_total += r["n_seg_gt"]
        n_seg_pred_total += r["n_seg_pred"]
        if r["seg_token_found"]:
            seg_emit_count += 1

        if r["n_seg_gt"] > 0:
            all_miou.append(r["per_image_miou"])
            by_task[r["task"]].append(r["per_image_miou"])
            by_camera[r["camera_type"]].append(r["per_image_miou"])

        gt_masks_per_image[img_id] = [_rle_to_mask(rle) for rle in r["gt_masks_rle"]]
        pred_masks_per_image[img_id] = [_rle_to_mask(rle) for rle in r["pred_masks_rle"]]
        gt_phrases_per_image[img_id] = r["gt_phrases"]
        pred_phrases_per_image[img_id] = r["pred_phrases"]
        gt_captions[img_id] = r["gt_caption"]
        pred_captions[img_id] = r["cleaned_caption"]

        gt_has_gas = bool(r.get("gt_gas_present", False))
        pred_has_gas = any(int(m.sum()) > 0 for m in pred_masks_per_image[img_id])
        if gt_has_gas and pred_has_gas:
            gas_det_tp += 1
        elif gt_has_gas and not pred_has_gas:
            gas_det_fn += 1
        elif (not gt_has_gas) and pred_has_gas:
            gas_det_fp += 1
        else:
            gas_det_tn += 1

        if r.get("task") == "gcg_analysis":
            if r.get("ratio_class_5") is not None:
                rc_gt.append(int(r["ratio_class_5"]))
                if r["pred_ratio"] >= 1:
                    rc_pred.append(int(r["pred_ratio"]))
                else:
                    rc_pred.append(0); rc_parse_fails += 1
            if r.get("ch4_pct") is not None:
                if r["pred_ch4"] is not None:
                    ch4_pairs.append((float(r["pred_ch4"]), float(r["ch4_pct"])))
                else:
                    ch4_parse_fails += 1
            if r.get("co2_pct") is not None:
                if r["pred_co2"] is not None:
                    co2_pairs.append((float(r["pred_co2"]), float(r["co2_pct"])))
                else:
                    co2_parse_fails += 1

    miou = float(np.mean(all_miou)) if all_miou else 0.0

    out = {
        "num_samples": num_total,
        "num_records_evaluated": len(records),
        "mIoU": miou,
        "seg_token_emission_rate": float(seg_emit_count / max(1, len(records))),
        "n_seg_gt_total": n_seg_gt_total,
        "n_seg_pred_total": n_seg_pred_total,
        "per_task_mIoU": {t: float(np.mean(v)) if v else 0.0 for t, v in by_task.items()},
        "per_camera_mIoU": {c: float(np.mean(v)) if v else 0.0 for c, v in by_camera.items()},
    }

    image_ids = sorted(gt_masks_per_image.keys())
    try:
        out["gcg_recall"] = _compute_gcg_recall(
            gt_phrases_per_image, pred_phrases_per_image,
            gt_masks_per_image, pred_masks_per_image,
            image_ids,
        )
    except Exception as e:
        out["gcg_recall_error"] = str(e)
    try:
        out["gcg_mask_ap"] = _compute_gcg_mask_ap(
            gt_masks_per_image, pred_masks_per_image, image_ids,
        )
    except Exception as e:
        out["gcg_mask_ap_error"] = str(e)
    try:
        out["captioning"] = _compute_gcg_captioning_metrics(
            gt_captions, pred_captions, image_ids,
        )
    except Exception as e:
        out["captioning_error"] = str(e)

    if rc_gt:
        rc_m = _ratio_class_metrics(rc_gt, rc_pred)
        rc_m["total"] = len(rc_gt)
        rc_m["parse_fail"] = rc_parse_fails
        rc_m["parse_fail_rate"] = rc_parse_fails / max(1, len(rc_gt))
        out["ratio_class"] = rc_m
        out["ratio_class_text"] = rc_m
    if ch4_pairs:
        ch4_m = _regression_metrics([p for p, _ in ch4_pairs], [g for _, g in ch4_pairs])
        ch4_m["samples"] = len(ch4_pairs); ch4_m["parse_fail"] = ch4_parse_fails
        out["ch4_concentration"] = ch4_m
        out["ch4_concentration_text"] = ch4_m
    if co2_pairs:
        co2_m = _regression_metrics([p for p, _ in co2_pairs], [g for _, g in co2_pairs])
        co2_m["samples"] = len(co2_pairs); co2_m["parse_fail"] = co2_parse_fails
        out["co2_concentration"] = co2_m
        out["co2_concentration_text"] = co2_m

    gas_total = gas_det_tp + gas_det_fp + gas_det_tn + gas_det_fn
    gas_acc = float((gas_det_tp + gas_det_tn) / max(1, gas_total))
    gas_prec = float(gas_det_tp / max(1, gas_det_tp + gas_det_fp))
    gas_rec = float(gas_det_tp / max(1, gas_det_tp + gas_det_fn))
    gas_f1 = float(2 * gas_prec * gas_rec / max(1e-10, gas_prec + gas_rec))
    out["gas_detection"] = {
        "tp": gas_det_tp, "fp": gas_det_fp, "tn": gas_det_tn, "fn": gas_det_fn,
        "accuracy": gas_acc, "precision": gas_prec, "recall": gas_rec, "f1": gas_f1,
        "total": gas_total,
    }

    vqa_from_text: dict = {}
    if "ratio_class" in out:
        vqa_from_text["ratio_class"] = out["ratio_class"]
    conc: dict = {}
    if "ch4_concentration" in out:
        c = out["ch4_concentration"]
        conc.update({"ch4_mae": c.get("mae"), "ch4_rmse": c.get("rmse"),
                     "ch4_r2": c.get("r2"), "ch4_mape": c.get("mape"),
                     "ch4_samples": c.get("samples")})
    if "co2_concentration" in out:
        c = out["co2_concentration"]
        conc.update({"co2_mae": c.get("mae"), "co2_rmse": c.get("rmse"),
                     "co2_r2": c.get("r2"), "co2_mape": c.get("mape"),
                     "co2_samples": c.get("samples")})
    if conc:
        vqa_from_text["concentration"] = conc
    if vqa_from_text:
        out["vqa_from_text"] = vqa_from_text

    return out


def eval_level3(model, tokenizer, seg_token_id, cfg, device, test_jsonl,
                max_samples, rank, world_size):
    if rank == 0:
        print("\n" + "=" * 60); print("LEVEL 3: GCG"); print("=" * 60)
    full_ds = VQADataset(
        jsonl_path=test_jsonl, tokenizer=tokenizer,
        image_size=cfg["data"]["img_size"], max_length=512, seg_token="[SEG]",
        temporal_mode=cfg["data"].get("temporal_mode", "single"),
    )
    indices = [i for i, r in enumerate(full_ds.records) if r.get("level") == 3]
    if max_samples > 0:
        indices = subsample(indices, max_samples)
    if rank == 0:
        print(f"Evaluating {len(indices)} L3 samples across {world_size} rank(s)")

    records = _run_level3(model, tokenizer, seg_token_id, full_ds, indices,
                          device, rank, world_size)
    if world_size > 1:
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or "local"
        tmpdir = f"/tmp/gasvlm_l3_gather_{job_id}"
        records = collect_results_cpu(records, len(indices), tmpdir=tmpdir)
        if rank != 0:
            return None
    return _metrics_level3(records, num_total=len(indices))


# --- Top-level evaluate() and main() ------------------------------------------

def _save_level_result(out_dir: str, name: str, data: dict, meta: dict):
    payload = dict(meta); payload[name] = data
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def evaluate(cfg: dict, ckpt: str, out_dir: str, levels: list[int],
             max_samples: int, split: str, rank: int, world_size: int):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if split == "test":
        jsonl = cfg["data"].get("test_jsonl", "data/test.jsonl")
    elif split == "val":
        jsonl = cfg["data"].get("val_jsonl", "data/val.jsonl")
    else:
        jsonl = cfg["data"].get("train_jsonl", "data/train.jsonl")

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Evaluating levels {levels} on {split} split")
        print(f"JSONL: {jsonl}")
        print(f"Output: {out_dir}")
        if max_samples > 0:
            print(f"Max samples per level: {max_samples}")

    model, tokenizer, seg_token_id, _ = build_model(cfg, ckpt, device)

    if not FAST_DECODE_OFF and device.type == "cuda":
        from gasvlm.fast_decode import enable_fast_decode
        enable_fast_decode(model)
        if rank == 0:
            print("Fast decode enabled (static KV cache + CUDA graphs); "
                  "first generate call compiles, ~1 min")

    meta = {"checkpoint": os.path.basename(ckpt), "split": split,
            "max_samples": max_samples}
    combined = dict(meta)
    t0 = time.time()

    if 1 in levels:
        t1 = time.time()
        l1 = eval_level1(model, tokenizer, seg_token_id, cfg, device,
                         jsonl, max_samples, rank, world_size)
        if rank == 0 and l1 is not None:
            l1["elapsed_seconds"] = round(time.time() - t1, 1)
            combined["level_1_segmentation"] = l1
            _save_level_result(out_dir, "level_1_segmentation", l1, meta)

    if 2 in levels:
        t2 = time.time()
        l2 = eval_level2(model, tokenizer, seg_token_id, cfg, device,
                         jsonl, max_samples, rank, world_size)
        if rank == 0 and l2 is not None:
            l2["elapsed_seconds"] = round(time.time() - t2, 1)
            combined["level_2_vqa"] = l2
            _save_level_result(out_dir, "level_2_vqa", l2, meta)

    if 3 in levels:
        t3 = time.time()
        l3 = eval_level3(model, tokenizer, seg_token_id, cfg, device,
                         jsonl, max_samples, rank, world_size)
        if rank == 0 and l3 is not None:
            l3["elapsed_seconds"] = round(time.time() - t3, 1)
            combined["level_3_gcg"] = l3
            _save_level_result(out_dir, "level_3_gcg", l3, meta)

    if rank == 0:
        elapsed = time.time() - t0
        combined["elapsed_seconds"] = round(elapsed, 1)
        out_path = os.path.join(out_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print(f"  Total: {elapsed/60:.1f} min")
        if "level_1_segmentation" in combined:
            o = combined["level_1_segmentation"]["overall"]
            pc = combined["level_1_segmentation"]["per_class"]
            print(f"  L1 gIoU={o['gIoU']:.4f} cIoU={o['cIoU']:.4f} "
                  f"mIoU3={o['mIoU_3class']:.4f} Dice={o['mean_Dice']:.4f}")
            print(f"     gas IoU={pc['gas']['IoU']:.4f} src IoU={pc['source']['IoU']:.4f}")
        if "level_2_vqa" in combined:
            l2 = combined["level_2_vqa"]
            if "ratio_class" in l2:
                gen = l2["ratio_class"].get("gen", {})
                print(f"  L2 ratio_acc={gen.get('accuracy',0):.4f} "
                      f"within1={gen.get('within_1_accuracy',0):.4f} "
                      f"parse_fail={gen.get('parse_fail_rate',0):.2%}")
            if "concentration" in l2:
                gen = l2["concentration"].get("gen", {})
                print(f"  L2 ch4 MAE={gen.get('ch4_mae',0):.2f}% "
                      f"(parse_fail={gen.get('ch4_parse_fail',0)})")
                print(f"  L2 co2 MAE={gen.get('co2_mae',0):.2f}% "
                      f"(parse_fail={gen.get('co2_parse_fail',0)})")
        if "level_3_gcg" in combined:
            l3 = combined["level_3_gcg"]
            print(f"  L3 mIoU(free-gen)={l3['mIoU']:.4f} "
                  f"seg_emit_rate={l3.get('seg_token_emission_rate',0):.2%} "
                  f"(n_seg_pred={l3['n_seg_pred_total']}/{l3['n_seg_gt_total']})")
            if "ratio_class_text" in l3:
                rc = l3["ratio_class_text"]
                print(f"  L3 ratio_acc(text)={rc['accuracy']:.4f} "
                      f"parse_fail={rc.get('parse_fail_rate',0):.2%}")
            for key, label in (("ch4_concentration_text", "ch4(text)"),
                                ("co2_concentration_text", "co2(text)")):
                if key in l3:
                    m = l3[key]; r2 = m.get('r2')
                    r2s = f"{r2:.4f}" if r2 is not None else "n/a"
                    print(f"  L3 {label} MAE={m.get('mae',0):.2f}% R²={r2s} "
                          f"(parse_fail={m.get('parse_fail',0)})")
        print(f"  Saved → {out_path}")
        print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="GasVLM per-level evaluation")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--levels", default="1,2,3", type=str)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--max-samples", default=0, type=int,
                    help="if >0, evaluate this many records per level, strided "
                         "evenly over the split (0 = full split)")
    ap.add_argument("--no-fast-decode", action="store_true",
                    help="use the eager dynamic-cache decode path instead of "
                         "the default static-cache + CUDA-graph path")
    ap.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    ap.add_argument("--local_rank", "--local-rank", default=0, type=int)
    args = ap.parse_args()

    global FAST_DECODE_OFF
    FAST_DECODE_OFF = args.no_fast_decode

    if args.launcher == "pytorch":
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        _init_dist_pytorch("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1

    cfg = load_config(Path(args.config))
    levels = [int(s) for s in args.levels.split(",")]
    evaluate(cfg, args.checkpoint, args.output_dir, levels,
             args.max_samples, args.split, rank, world_size)


if __name__ == "__main__":
    main()
