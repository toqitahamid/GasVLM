# Data format

GasVLM trains and evaluates on JSONL files of grounded vision-language records
over optical-gas-imaging (OGI) thermal frames. The dataset itself is not part of
this repository. GasGround is on the Hugging Face Hub at
[toqi/GasGround](https://huggingface.co/datasets/toqi/GasGround); access is
granted after you accept its terms. This document specifies the record contract,
so you can also point the code at your own data.

The config (`configs/gasvlm_headline.yaml`) expects three files:

```
data/train.jsonl
data/val.jsonl
data/test.jsonl
```

Relative `image_path` / `mask_path` are resolved against the repository root.

## Record schema

One JSON object per line:

| Field           | Type        | Required | Notes |
|-----------------|-------------|----------|-------|
| `image_path`    | str         | yes      | Path to a thermal frame (any format OpenCV reads; loaded grayscale, replicated to 3 channels). |
| `question`      | str         | yes      | The user turn. |
| `answer`        | str         | yes      | The target answer. May contain `[SEG]` tokens and `<p>...</p>` grounded-phrase markup. |
| `mask_path`     | str         | no       | Grayscale PNG mask (see convention below). Required for records that contain `[SEG]`. |
| `mask_classes`  | list[str]   | no       | One class name per `[SEG]`, in order. Each maps to a pixel value (below). |
| `level`         | int         | yes      | Task level: `1` = RES (segmentation), `2` = VQA, `3` = GCG. Drives sampling and per-level eval. |
| `task`          | str         | no       | Free-form task tag, used only for per-task eval breakdowns. |
| `ratio_class_5` | int (1..5)  | no       | Coarse CH4/CO2 mixture class; `-1` if unknown. |
| `ch4_pct`       | float       | no       | Methane percentage; `-1.0` if unknown. |
| `co2_pct`       | float       | no       | CO2 percentage; `-1.0` if unknown. |
| `camera_type`   | str         | no       | `"co2"` or `"methane"`. Used only for per-camera eval breakdowns; the model does not condition on it. Defaults to `"co2"`. |
| `cot_answer`    | str         | no       | Optional chain-of-thought answer, used only when `use_cot: true`. |

## Mask convention

Masks are single-channel (grayscale) PNGs with three pixel values:

| Pixel value | Class name (`mask_classes`) | Meaning |
|-------------|-----------------------------|---------|
| `0`         | (background)                | Not segmented. |
| `100`       | `source`                    | Gas-release source (module outlet). |
| `255`       | `gas`                       | Gas plume. |

`mask_classes` selects which value becomes the positive class for each `[SEG]`,
in the order the `[SEG]` tokens appear in `answer`. For example a GCG answer with
`<p>the plume</p>[SEG]` and `<p>the source</p>[SEG]` uses
`"mask_classes": ["gas", "source"]`.

## Example record (RES / level 1)

```json
{
  "image_path": "data/frames/methane_clip_03/frame_00120.png",
  "mask_path": "data/masks/methane_clip_03/frame_00120.png",
  "mask_classes": ["gas"],
  "question": "Segment the gas plume.",
  "answer": "Sure, the gas plume is [SEG].",
  "level": 1,
  "task": "res_plume",
  "ratio_class_5": 3,
  "ch4_pct": 31.0,
  "co2_pct": 69.0,
  "camera_type": "methane"
}
```
