"""GasVLM dataset: a thin wrapper over :class:`VQADataset`.

Adds ``camera_type`` passthrough (so per-camera eval breakdowns work, even
though the single-LoRA encoder does not condition on it), optional thermal
augmentation, and optional chain-of-thought answers (the ``cot_answer`` field
when present). Default ``image_size=256`` matches V-JEPA 2 ViT-L's crop size.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from gasvlm.data.vqa_dataset import VQADataset, vqa_collate_fn


class GasVLMDataset(VQADataset):
    """VQADataset + camera_type passthrough + thermal augmentation + CoT answers.

    Args:
        Same as VQADataset, plus:
        augment: Optional thermal augmentation callable(img, mask) → (img, mask).
                 Applied during training only. Image must be float32 [0,1].
        use_cot: If True, prefer ``cot_answer`` field over ``answer``.
    """

    def __init__(self, *args, augment=None, use_cot: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.use_cot = use_cot
        # consecutive_pair + augment requires applying identical geometric
        # transforms to both frames so inter-frame motion isn't broken. That
        # composition is intentionally out of scope for the F1 patch; gate it.
        if self.temporal_mode == "consecutive_pair" and self.augment is not None:
            raise NotImplementedError(
                "temporal_mode='consecutive_pair' with thermal augmentation is "
                "not supported yet: augmentation must be applied identically to "
                "both frames to preserve inter-frame motion. Disable "
                "use_thermal_augment for F1, or extend ThermalAugment to take "
                "a frame stack."
            )

    def __getitem__(self, idx: int) -> dict:
        if self._sample_rates is not None:
            import numpy as np
            level = np.random.choice(self._levels, p=self._sample_rates)
            indices = self._level_indices[level]
            idx = indices[np.random.randint(len(indices))]
        record = self.records[idx]

        # --- Image preprocessing ---
        import cv2
        import numpy as np

        img = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {record['image_path']}")

        # --- Mask loading (before augmentation, for geometric consistency) ---
        mask_classes = record.get("mask_classes", [])
        mask_path = record.get("mask_path")
        mask_raw = None
        if mask_path is not None and Path(mask_path).exists() and mask_classes:
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert image to float32 [0,1] and replicate to 3ch
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0

        # Apply augmentation if provided (training mode)
        if self.augment is not None:
            aug_img, aug_mask = self.augment(img, mask_raw)
            img = aug_img
            if aug_mask is not None:
                mask_raw = aug_mask

        # Resize to target size (if not already resized by augment)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            if mask_raw is not None:
                mask_raw = cv2.resize(
                    mask_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST,
                )

        # Normalize (ImageNet stats)
        img = (img - self.MEAN) / self.STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        # Stack the predecessor frame at index 0 for consecutive_pair mode.
        # Augmentation has already been verified disabled in __init__, so it is
        # safe to load the predecessor via the parent's unaugmented helper:
        # both frames then go through the same resize + ImageNet normalization.
        if self.temporal_mode == "consecutive_pair":
            self._n_prev_lookups += 1
            prev_path = self._resolve_prev_image_path(record["image_path"])
            prev_tensor = None
            if prev_path is not None:
                try:
                    prev_tensor = self._load_frame_tensor(str(prev_path))
                except FileNotFoundError:
                    prev_tensor = None
            if prev_tensor is None:
                self._n_prev_missing += 1
                prev_tensor = img_tensor  # duplicate-frame fallback
            img_tensor = torch.stack([prev_tensor, img_tensor], dim=0)  # [2, 3, H, W]

        # Extract binary masks
        gt_masks = []
        if mask_raw is not None and mask_classes:
            from gasvlm.data.vqa_dataset import _extract_binary_masks
            for binary_np in _extract_binary_masks(mask_raw, mask_classes):
                gt_masks.append(torch.from_numpy(binary_np))

        # --- Tokenization ---
        question = record["question"]
        answer = record.get("cot_answer", record["answer"]) if self.use_cot else record["answer"]

        if self._use_chat_template:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": question})

            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=False,
                enable_thinking=False,
            )
            answer_enc = self.tokenizer(
                answer, add_special_tokens=False, return_attention_mask=False,
            )
            eos_id = self.tokenizer.eos_token_id
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            end_token = eot_id if eot_id != self.tokenizer.unk_token_id else eos_id
            answer_ids = answer_enc["input_ids"] + [end_token]
            question_ids = prompt_ids
        else:
            question_enc = self.tokenizer(
                question, add_special_tokens=True, return_attention_mask=False,
            )
            answer_enc = self.tokenizer(
                answer, add_special_tokens=False, return_attention_mask=False,
            )
            question_ids = question_enc["input_ids"]
            answer_ids = answer_enc["input_ids"] + [self.tokenizer.eos_token_id]

        combined_ids = question_ids + answer_ids
        if len(combined_ids) > self.max_length:
            combined_ids = combined_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            q_len = min(len(question_ids), self.max_length)
        else:
            q_len = len(question_ids)

        labels = [self.IGNORE_INDEX] * q_len + combined_ids[q_len:]
        seg_positions = [tid == self._seg_token_id for tid in combined_ids]

        input_ids = torch.tensor(combined_ids, dtype=torch.long)
        attention_mask = torch.ones(len(combined_ids), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        seg_positions = torch.tensor(seg_positions, dtype=torch.bool)

        return {
            "image": img_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seg_positions": seg_positions,
            "gt_masks": gt_masks,
            "level": record.get("level", 0),
            "task": record.get("task", ""),
            "ratio_class": record.get("ratio_class_5", -1),
            "ch4_pct": float(record.get("ch4_pct", -1.0)),
            "co2_pct": float(record.get("co2_pct", -1.0)),
            "q_len": q_len,
            "camera_type": record.get("camera_type", "co2"),
        }


def _collate(batch, *, pad_token_id: int):
    out = vqa_collate_fn(batch, pad_token_id=pad_token_id)
    out["camera_type"] = [s["camera_type"] for s in batch]
    return out


def build_loader(
    jsonl_path: str,
    tokenizer,
    image_size: int = 256,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 8,
    sample_rates: Optional[dict] = None,
    shuffle: bool = True,
    distributed: bool = False,
    drop_last: bool = True,
    augment=None,
    use_cot: bool = False,
    temporal_mode: str = "single",
):
    ds = GasVLMDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=max_length,
        seg_token="[SEG]",
        sample_rates=sample_rates,
        augment=augment,
        use_cot=use_cot,
        temporal_mode=temporal_mode,
    )
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=shuffle, drop_last=drop_last
        )
        shuffle = False
    collate = partial(_collate, pad_token_id=int(tokenizer.pad_token_id or 0))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate,
    )
    return loader, sampler
