"""Pure helper functions for GasGround VQA evaluation.

No model imports here; everything is importable and unit-testable in isolation.

Sections:
    A. Concentration parsing
    B. Regression metrics (MAE, RMSE, MAPE, R²)
    C. Ratio-class metrics (accuracy, within-1, macro-F1, per-class F1)
    D. GCG evaluation (mask IoU, captioning, mask AP, grounded recall)
"""

import json
import os
import re
import tempfile

import numpy as np

# Number of ratio classes (v2: 5 classes for ratio_class_5)
NUM_RATIO_CLASSES = 5

# ---------------------------------------------------------------------------
# A. Concentration parsing
# ---------------------------------------------------------------------------

_CH4_PATTERNS = [
    # 1. "X% methane" / "X% CH4"
    re.compile(r'(\d+\.?\d*)\s*%\s*(?:methane|CH4|ch4)', re.IGNORECASE),
    # 2. "methane ... X%" (wider gap: 50 non-digit chars vs original 20)
    re.compile(r'(?:methane|CH4|ch4)[^\d]{0,50}?(\d+\.?\d*)\s*%', re.IGNORECASE),
    # 3. "X percent methane" / "X pct CH4"
    re.compile(r'(\d+\.?\d*)\s*(?:percent|pct)\b[^\d]{0,30}?(?:methane|CH4|ch4)', re.IGNORECASE),
    # 4. "methane ... X percent"
    re.compile(r'(?:methane|CH4|ch4)[^\d]{0,50}?(\d+\.?\d*)\s*(?:percent|pct)\b', re.IGNORECASE),
]
_CO2_PATTERNS = [
    re.compile(r'(\d+\.?\d*)\s*%\s*(?:carbon\s*dioxide|CO2|co2)', re.IGNORECASE),
    re.compile(r'(?:carbon\s*dioxide|CO2|co2)[^\d]{0,50}?(\d+\.?\d*)\s*%', re.IGNORECASE),
    re.compile(r'(\d+\.?\d*)\s*(?:percent|pct)\b[^\d]{0,30}?(?:carbon\s*dioxide|CO2|co2)', re.IGNORECASE),
    re.compile(r'(?:carbon\s*dioxide|CO2|co2)[^\d]{0,50}?(\d+\.?\d*)\s*(?:percent|pct)\b', re.IGNORECASE),
]


def _parse_concentrations(text: str) -> tuple:
    """Parse CH4% and CO2% from model output text.

    Returns:
        (ch4_val or None, co2_val or None)
    """
    ch4_val = None
    for pat in _CH4_PATTERNS:
        m = pat.search(text)
        if m:
            ch4_val = float(m.group(1))
            break

    co2_val = None
    for pat in _CO2_PATTERNS:
        m = pat.search(text)
        if m:
            co2_val = float(m.group(1))
            break

    return ch4_val, co2_val


# ---------------------------------------------------------------------------
# B. Regression metrics
# ---------------------------------------------------------------------------

def _regression_metrics(preds: list, gts: list) -> dict:
    """Compute MAE, RMSE, R², MAPE from prediction and ground-truth lists."""
    p, g = np.array(preds, dtype=float), np.array(gts, dtype=float)
    mae = float(np.mean(np.abs(p - g)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mape = float(np.mean(np.abs(p - g) / np.maximum(g, 1e-10)) * 100)
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    # R² is undefined when GT has zero variance (all samples share one value).
    # The +1e-10 guard from earlier versions produced huge negative numbers
    # that looked like a real fit failure — return None instead so callers
    # can tell "too few / too similar samples" apart from "bad predictions".
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else None
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "samples": len(preds)}


# ---------------------------------------------------------------------------
# C. Ratio-class metrics
# ---------------------------------------------------------------------------

_RATIO_CLASS_PATTERNS = [
    # Original: "Class N" / "class N"
    re.compile(r'[Cc]lass\s*(\d{1,2})\b'),
    # NEW: "category N", "tier N", "ratio N", "ratio class N", "ratio_class N"
    re.compile(r'(?:category|tier|ratio[_\s]+class|ratio)\s*(?:is\s*|=\s*|:\s*)?(\d{1,2})\b', re.IGNORECASE),
    # NEW: "N of 5" / "N out of 5" — the GT format includes "Class 3 of 5"; some models drop the "Class" prefix
    re.compile(r'\b(\d{1,2})\s*(?:of|out\s*of)\s*5\b', re.IGNORECASE),
]


def _parse_ratio_class(text: str) -> int:
    """Extract ratio class (1–NUM_RATIO_CLASSES) from generated text. Returns -1 on failure."""
    for pat in _RATIO_CLASS_PATTERNS:
        m = pat.search(text)
        if m:
            cls = int(m.group(1))
            if 1 <= cls <= NUM_RATIO_CLASSES:
                return cls
    return -1


def _ratio_class_metrics(gt_classes: list, pred_classes: list) -> dict:
    """Compute accuracy, within-1 accuracy, macro-F1, per-class F1, and confusion matrix.

    Macro-F1 follows sklearn convention (average='macro', zero_division=0):
    averaged only over classes that have at least one GT sample.
    """
    correct = 0
    within_1 = 0
    confusion = np.zeros((NUM_RATIO_CLASSES, NUM_RATIO_CLASSES), dtype=int)

    for gt, pred in zip(gt_classes, pred_classes):
        if pred == gt:
            correct += 1
        if abs(pred - gt) <= 1:
            within_1 += 1
        if 1 <= gt <= NUM_RATIO_CLASSES and 1 <= pred <= NUM_RATIO_CLASSES:
            confusion[gt - 1][pred - 1] += 1

    total = len(gt_classes)
    acc = correct / max(total, 1)
    w1_acc = within_1 / max(total, 1)

    macro_f1 = 0.0
    n_active = 0
    per_class_f1 = {}
    for c in range(NUM_RATIO_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        if confusion[c, :].sum() > 0:
            macro_f1 += f1
            n_active += 1
            per_class_f1[c + 1] = float(f1)
    macro_f1 = float(macro_f1 / max(n_active, 1))

    return {
        "accuracy": float(acc),
        "within_1_accuracy": float(w1_acc),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------------
# D. GCG evaluation helpers
# ---------------------------------------------------------------------------

def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _compute_miou_one_to_one(pred_masks_np: list, gt_masks_np: list) -> float:
    """Compute mIoU with greedy one-to-one matching (GLaMM style).

    Args:
        pred_masks_np: list of binary numpy arrays [H, W]
        gt_masks_np:   list of binary numpy arrays [H, W]
    Returns:
        Mean IoU of matched pairs, or 0.0 if no pairs.
    """
    if not pred_masks_np or not gt_masks_np:
        return 0.0
    iou_matrix = np.zeros((len(pred_masks_np), len(gt_masks_np)))
    for i, pm in enumerate(pred_masks_np):
        for j, gm in enumerate(gt_masks_np):
            iou_matrix[i, j] = _compute_iou(pm, gm)
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        paired_iou.append(iou_matrix[idx])
        iou_matrix = np.delete(iou_matrix, idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, idx[1], axis=1)
    return float(np.mean(paired_iou)) if paired_iou else 0.0


def _compute_gcg_captioning_metrics(gt_captions: dict, pred_captions: dict,
                                    image_ids: list) -> dict:
    """Compute BLEU, METEOR, ROUGE_L, CIDEr, SPICE using pycocoevalcap.

    Args:
        gt_captions:   {image_id: caption_string}
        pred_captions: {image_id: caption_string}
        image_ids:     list of image_id ints
    Returns:
        dict of metric_name → score
    """
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    gt_data = {
        "images": [{"id": iid} for iid in image_ids],
        "annotations": [
            {"image_id": iid, "id": idx, "caption": gt_captions[iid]}
            for idx, iid in enumerate(image_ids)
        ],
    }
    pred_data = [
        {"image_id": iid, "caption": pred_captions.get(iid, "")}
        for iid in image_ids
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_data, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_data, f)
        pred_path = f.name

    try:
        coco_gt = COCO(gt_path)
        coco_res = coco_gt.loadRes(pred_path)
        coco_eval = COCOEvalCap(coco_gt, coco_res)
        coco_eval.params["image_id"] = image_ids
        try:
            coco_eval.evaluate()
        except Exception as e:
            # SPICE often fails (Java). BLEU/METEOR/ROUGE/CIDEr are computed first.
            print(f"  Warning: captioning scorer partial failure (likely SPICE): {e}")
        return dict(coco_eval.eval) if coco_eval.eval else {}
    except Exception as e:
        print(f"  Warning: captioning metrics setup failed: {e}")
        return {}
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)


def _compute_gcg_mask_ap(gt_masks_per_image: dict, pred_masks_per_image: dict,
                         image_ids: list) -> dict:
    """Compute COCO segmentation AP/AP50/AP75 using pycocotools.

    Args:
        gt_masks_per_image:   {image_id: [np.array, …]}
        pred_masks_per_image: {image_id: [np.array, …]}
        image_ids:            list of image_id ints
    Returns:
        dict with keys AP, AP50, AP75
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskUtils

    gt_anns = []
    ann_id = 1
    images_meta = []
    for iid in image_ids:
        masks = gt_masks_per_image.get(iid, [])
        if masks:
            h, w = masks[0].shape
        else:
            p_masks = pred_masks_per_image.get(iid, [])
            h, w = p_masks[0].shape if p_masks else (512, 640)
        images_meta.append({"id": iid, "height": h, "width": w})
        for m in masks:
            rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            gt_anns.append({
                "id": ann_id, "image_id": iid, "category_id": 1,
                "segmentation": rle, "area": int(m.sum()),
                "iscrowd": 0, "bbox": list(maskUtils.toBbox(rle).astype(float)),
            })
            ann_id += 1

    gt_data = {
        "images": images_meta,
        "annotations": gt_anns,
        "categories": [{"id": 1, "name": "object"}],
    }

    pred_anns = []
    for iid in image_ids:
        for m in pred_masks_per_image.get(iid, []):
            rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            pred_anns.append({
                "image_id": iid, "category_id": 1,
                "segmentation": rle, "score": 1.0,
            })

    if not gt_anns or not pred_anns:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_data, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_anns, f)
        pred_path = f.name

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
        }
    except Exception as e:
        print(f"  Warning: mask AP failed: {e}")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)


def _compute_gcg_recall(gt_phrases_per_image: dict, pred_phrases_per_image: dict,
                        gt_masks_per_image: dict, pred_masks_per_image: dict,
                        image_ids: list,
                        iou_threshold: float = 0.5,
                        text_sim_threshold: float = 0.5) -> float:
    """Compute Recall@IoU+TextSim with BERT embeddings (GLaMM style).

    Returns recall float.
    """
    import torch
    from transformers import AutoTokenizer as BertTokenizer, AutoModel as BertModel
    from sklearn.metrics.pairwise import cosine_similarity

    print("  Loading BERT for text similarity...")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    # Phrases are heavily templated (GasGround: a handful of unique strings
    # across ~10k records) — memoize or this loop is hours of CPU BERT.
    _emb_cache: dict = {}

    def _embed(text):
        if text in _emb_cache:
            return _emb_cache[text]
        inputs = bert_tok(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = bert_model(**inputs)
        _emb_cache[text] = out.last_hidden_state[0].mean(dim=0).numpy()
        return _emb_cache[text]

    true_positives = 0
    actual_positives = 0

    for iid in image_ids:
        gt_phrases = gt_phrases_per_image.get(iid, [])
        pred_phrases = pred_phrases_per_image.get(iid, [])
        gt_masks = gt_masks_per_image.get(iid, [])
        pred_masks = pred_masks_per_image.get(iid, [])

        actual_positives += len(gt_phrases)
        if not gt_phrases or not pred_phrases or not gt_masks or not pred_masks:
            continue

        # Clip predicted items to the shorter of masks/phrases — generation can
        # generate a different number of each, which would give iou_mat and
        # text_sims different column counts and crash the greedy loop.
        n_pred = min(len(pred_masks), len(pred_phrases))
        pred_masks = pred_masks[:n_pred]
        pred_phrases = pred_phrases[:n_pred]

        iou_mat = np.zeros((len(gt_masks), n_pred))
        for i, gm in enumerate(gt_masks):
            for j, pm in enumerate(pred_masks):
                iou_mat[i, j] = _compute_iou(gm, pm)

        gt_embs = [_embed(p) for p in gt_phrases]
        pred_embs = [_embed(p) for p in pred_phrases]
        text_sims = np.zeros((len(gt_phrases), n_pred))
        for i, ge in enumerate(gt_embs):
            for j, pe in enumerate(pred_embs):
                text_sims[i, j] = cosine_similarity([ge], [pe])[0, 0]

        while iou_mat.size > 0:
            idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            # Greedy match by max IoU. Matrix is sorted-by-max: once the best
            # remaining IoU falls below threshold, no further pair can qualify.
            if iou_mat[idx] < iou_threshold:
                break
            if text_sims[idx] >= text_sim_threshold:
                true_positives += 1
            # Always consume the pair (matched or not) so the next iteration
            # considers a different GT/pred pairing rather than re-picking this one.
            iou_mat[idx[0], :] = 0
            iou_mat[:, idx[1]] = 0
            text_sims[idx[0], :] = 0
            text_sims[:, idx[1]] = 0

    return true_positives / actual_positives if actual_positives > 0 else 0.0
