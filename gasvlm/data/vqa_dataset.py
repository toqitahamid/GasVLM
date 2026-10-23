"""
VQA Dataset loader for grounded visual question answering on OGI imagery.

Reads v2 JSONL files produced by scripts/data/generate_vqa_v2_vllm.py. Each line contains:
    - question/answer text (with optional <p>phrase</p>[SEG] markup)
    - image_path to thermal frame
    - mask_path to segmentation mask (optional)
    - ratio metadata (ch4_pct, co2_pct, ratio_class_5, level)

Handles I-JEPA preprocessing (replicate grayscale to 3ch, normalize, resize 224x224).
Tokenizes question+answer with causal masking (question tokens ignored in loss).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Mask class name → raw pixel value in mask PNGs
_CLASS_TO_PIXEL = {"gas": 255, "source": 100}


def _extract_binary_masks(
    mask_raw: np.ndarray, mask_classes: list[str],
) -> list[np.ndarray]:
    """Extract per-class binary masks from a multi-class mask PNG.

    Args:
        mask_raw: Grayscale mask with values 0=bg, 100=source, 255=gas.
        mask_classes: Ordered list of class names, one per [SEG] token.

    Returns:
        List of binary float32 masks, one per mask_class entry.
    """
    masks = []
    for cls_name in mask_classes:
        pixel_val = _CLASS_TO_PIXEL.get(cls_name, 255)
        masks.append((mask_raw == pixel_val).astype(np.float32))
    return masks


class VQADataset(Dataset):
    """PyTorch dataset for VQA training on OGI thermal gas imagery.

    Each sample contains:
        - image: [3, H, W] normalized thermal image (grayscale replicated to
          3ch) when ``temporal_mode='single'`` (default); or
          [2, 3, H, W] with the predecessor frame stacked at index 0 and the
          current frame at index 1 when ``temporal_mode='consecutive_pair'``.
          The collator preserves the rank by stacking along a new batch dim, so
          downstream consumers see ``[B, 3, H, W]`` or ``[B, 2, 3, H, W]``.
        - input_ids: [L] tokenized question + answer
        - attention_mask: [L] attention mask (1 = attend, 0 = padding)
        - labels: [L] target token IDs (question tokens = -100 for ignore)
        - seg_positions: [L] bool tensor marking [SEG] token positions
        - gt_masks: list of [H, W] binary float tensors, one per [SEG] token

    When the tokenizer has a chat_template (instruct models), the question is
    wrapped as a user turn and the answer as an assistant turn, matching the
    format the LLM was instruction-tuned with. Otherwise plain concatenation
    is used (base models).

    Args:
        jsonl_path: Path to JSONL file with VQA records
        tokenizer: HuggingFace tokenizer instance
        image_size: Target image size (square), default 224
        max_length: Maximum token sequence length
        seg_token: Special token marking segmentation output
        system_prompt: Optional system prompt for instruct models
    """

    # ImageNet normalization (I-JEPA expects this)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    IGNORE_INDEX = -100

    # Matches `{video_id}_frame_{NNNN}.{ext}` filenames so the predecessor frame
    # can be resolved by decrementing the numeric segment while preserving the
    # original zero-padding width (mixed 4-digit and 5-digit widths exist in
    # this dataset, but width is consistent within a single video).
    _FRAME_NUM_PATTERN = re.compile(
        r"^(?P<stem>.+_frame_)(?P<num>\d+)(?P<ext>\.[A-Za-z0-9]+)$"
    )

    @classmethod
    def _resolve_prev_image_path(cls, image_path: str) -> Optional[Path]:
        """Return the predecessor frame path, or None if it cannot be resolved.

        Predecessor = the same `{video_id}_frame_{N-1}.{ext}` in the same
        directory, with the original zero-padding width preserved. Returns
        ``None`` when:
          - the filename does not match the `..._frame_NNNN.{ext}` pattern,
          - the current frame is index 0 (no predecessor exists), or
          - the candidate predecessor file is absent from disk.
        Callers should treat ``None`` as a signal to fall back to duplicating
        the current frame so missing-predecessor records remain comparable to
        the single-frame headline.
        """
        p = Path(image_path)
        m = cls._FRAME_NUM_PATTERN.match(p.name)
        if m is None:
            return None
        num = int(m.group("num"))
        if num == 0:
            return None
        width = len(m.group("num"))
        prev_name = f"{m.group('stem')}{num - 1:0{width}d}{m.group('ext')}"
        prev_path = p.parent / prev_name
        return prev_path if prev_path.exists() else None

    DEFAULT_SYSTEM_PROMPT = (
        "You are a vision-language assistant specialized in optical gas imaging. "
        "You analyze thermal infrared images to detect gas plumes, identify gas "
        "release sources, segment regions of interest, and estimate methane/CO2 "
        "mixture ratios and concentrations."
    )

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        image_size: int = 224,
        max_length: int = 512,
        seg_token: str = "[SEG]",
        system_prompt: Optional[str] = None,
        sample_rates: Optional[dict[int, float]] = None,
        temporal_mode: str = "single",
    ):
        super().__init__()
        if temporal_mode not in ("single", "consecutive_pair"):
            raise ValueError(
                f"temporal_mode must be 'single' or 'consecutive_pair', got {temporal_mode!r}"
            )
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.seg_token = seg_token
        self.temporal_mode = temporal_mode
        # Main-process diagnostics for the consecutive_pair fallback rate;
        # not aggregated across DataLoader workers, so meaningful only when
        # iterating the dataset directly (num_workers=0).
        self._n_prev_lookups = 0
        self._n_prev_missing = 0

        self._use_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Load all records from JSONL
        self.records: list[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Task-aware sampling: group records by level (LISA-style probabilistic selection)
        self._sample_rates = None
        self._level_indices: dict[int, list[int]] = defaultdict(list)
        if sample_rates:
            for i, rec in enumerate(self.records):
                self._level_indices[rec.get("level", 0)].append(i)
            levels = sorted(sample_rates.keys())
            rates = np.array([sample_rates[lv] for lv in levels], dtype=np.float64)
            rates /= rates.sum()
            self._levels = levels
            self._sample_rates = rates

        # Ensure [SEG] token is in the tokenizer vocabulary
        if seg_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([seg_token], special_tokens=True)

        self._seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)

    def __len__(self) -> int:
        return len(self.records)

    def _load_frame_tensor(self, image_path: str) -> torch.Tensor:
        """Read a thermal frame from disk and apply the model's image preprocessing.

        Returns a ``[3, H, W]`` float tensor (ImageNet-normalized, grayscale
        replicated to 3 channels, resized to ``self.image_size``). Used for both
        the current frame and, in ``consecutive_pair`` mode, its predecessor.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._sample_rates is not None:
            level = np.random.choice(self._levels, p=self._sample_rates)
            indices = self._level_indices[level]
            idx = indices[np.random.randint(len(indices))]
        record = self.records[idx]

        # --- Image preprocessing ---
        img_tensor = self._load_frame_tensor(record["image_path"])  # [3, H, W]

        if self.temporal_mode == "consecutive_pair":
            self._n_prev_lookups += 1
            prev_path = self._resolve_prev_image_path(record["image_path"])
            prev_tensor: Optional[torch.Tensor] = None
            if prev_path is not None:
                try:
                    prev_tensor = self._load_frame_tensor(str(prev_path))
                except FileNotFoundError:
                    prev_tensor = None
            if prev_tensor is None:
                # Fallback: duplicate the current frame so missing-predecessor
                # records remain comparable to the single-frame headline.
                self._n_prev_missing += 1
                prev_tensor = img_tensor
            img_tensor = torch.stack([prev_tensor, img_tensor], dim=0)  # [2, 3, H, W]

        # --- Mask loading → pre-separated binary masks (LISA/GLaMM convention) ---
        mask_classes = record.get("mask_classes", [])
        mask_path = record.get("mask_path")
        gt_masks: list[torch.Tensor] = []

        if mask_path is not None and Path(mask_path).exists() and mask_classes:
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_raw = cv2.resize(
                mask_raw,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            for binary_np in _extract_binary_masks(mask_raw, mask_classes):
                gt_masks.append(torch.from_numpy(binary_np))  # [H, W] float32

        # --- Tokenization ---
        question = record["question"]
        answer = record["answer"]

        if self._use_chat_template:
            # Build prompt (everything up to the assistant's reply) using the
            # tokenizer's chat_template so the instruct model sees the format
            # it was trained with.
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": question})

            # Prompt = system + user turn + generation header (no answer yet)
            prompt_ids: list[int] = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=False,
            )

            # Tokenize the answer separately (no special tokens — those come
            # from the chat template).  Append the model's EOS / end-of-turn
            # token so the model learns to stop after answering.
            answer_enc = self.tokenizer(
                answer, add_special_tokens=False, return_attention_mask=False,
            )
            eos_id = self.tokenizer.eos_token_id
            # Llama-3 Instruct uses <|eot_id|> to end a turn
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            end_token = eot_id if eot_id != self.tokenizer.unk_token_id else eos_id
            answer_ids = answer_enc["input_ids"] + [end_token]

            question_ids = prompt_ids
        else:
            # Base model fallback: plain concatenation
            question_enc = self.tokenizer(
                question, add_special_tokens=True, return_attention_mask=False,
            )
            answer_enc = self.tokenizer(
                answer, add_special_tokens=False, return_attention_mask=False,
            )
            question_ids = question_enc["input_ids"]
            answer_ids = answer_enc["input_ids"] + [self.tokenizer.eos_token_id]

        # Concatenate and truncate to max_length
        combined_ids = question_ids + answer_ids
        if len(combined_ids) > self.max_length:
            combined_ids = combined_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            q_len = min(len(question_ids), self.max_length)
        else:
            q_len = len(question_ids)

        # Build labels: prompt/question tokens are ignored, answer tokens are targets
        labels = [self.IGNORE_INDEX] * q_len + combined_ids[q_len:]
        assert len(labels) == len(combined_ids)

        # Find [SEG] token positions
        seg_positions = [tid == self._seg_token_id for tid in combined_ids]

        # Convert to tensors
        input_ids = torch.tensor(combined_ids, dtype=torch.long)
        attention_mask = torch.ones(len(combined_ids), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        seg_positions = torch.tensor(seg_positions, dtype=torch.bool)

        result = {
            "image": img_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seg_positions": seg_positions,
            "gt_masks": gt_masks,
            # Metadata for per-level/per-task evaluation
            "level": record.get("level", 0),
            "task": record.get("task", ""),
            "ratio_class": record.get("ratio_class_5", -1),
            "ch4_pct": float(record.get("ch4_pct", -1.0)),
            "co2_pct": float(record.get("co2_pct", -1.0)),
            "q_len": q_len,
        }

        return result


def vqa_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate VQA samples into a batch with padding.

    Pads input_ids, attention_mask, labels, and seg_positions to the
    maximum sequence length in the batch. Stacks images and masks directly.

    Args:
        batch: List of sample dicts from VQADataset.__getitem__
        pad_token_id: Value for padded input_ids positions; must match
            ``tokenizer.pad_token_id`` in training (LISA/LLaVA-style).

    Returns:
        Batched dict with padded tensors
    """
    max_len = max(sample["input_ids"].size(0) for sample in batch)

    images = []
    gt_masks_list = []
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    seg_positions_list = []

    for sample in batch:
        images.append(sample["image"])
        gt_masks_list.append(sample.get("gt_masks", []))

        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        pad_tail = torch.full((pad_len,), pad_token_id, dtype=torch.long)
        input_ids_list.append(torch.cat([sample["input_ids"], pad_tail]))
        # Pad attention_mask with 0 (ignore padded positions)
        attention_mask_list.append(
            torch.cat([sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        # Pad labels with -100 (ignored in loss)
        labels_list.append(
            torch.cat([
                sample["labels"],
                torch.full((pad_len,), VQADataset.IGNORE_INDEX, dtype=torch.long),
            ])
        )
        # Pad seg_positions with False
        seg_positions_list.append(
            torch.cat([
                sample["seg_positions"],
                torch.zeros(pad_len, dtype=torch.bool),
            ])
        )

    # Metadata (scalars/strings, not padded)
    levels = torch.tensor([s["level"] for s in batch], dtype=torch.long)
    ratio_classes = torch.tensor([s["ratio_class"] for s in batch], dtype=torch.long)
    ch4_pcts = torch.tensor([s["ch4_pct"] for s in batch], dtype=torch.float)
    co2_pcts = torch.tensor([s["co2_pct"] for s in batch], dtype=torch.float)
    q_lens = torch.tensor([s["q_len"] for s in batch], dtype=torch.long)
    tasks = [s["task"] for s in batch]

    return {
        "image": torch.stack(images),                    # [B, 3, H, W]
        "gt_masks": gt_masks_list,                       # list[list[Tensor]]
        "input_ids": torch.stack(input_ids_list),        # [B, L]
        "attention_mask": torch.stack(attention_mask_list),  # [B, L]
        "labels": torch.stack(labels_list),              # [B, L]
        "seg_positions": torch.stack(seg_positions_list),  # [B, L]
        "level": levels,                                 # [B]
        "task": tasks,                                   # list[str]
        "ratio_class": ratio_classes,                    # [B]
        "ch4_pct": ch4_pcts,                             # [B]
        "co2_pct": co2_pcts,                             # [B]
        "q_len": q_lens,                                 # [B]
    }
