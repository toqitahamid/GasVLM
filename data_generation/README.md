# GasGround question-answer generation

This folder holds the code that generated the question-answer records of
[GasGround](https://huggingface.co/datasets/toqi/GasGround). The prompts and
validator rules are the ones used to build the released dataset.

## Files

| File | Contents |
|---|---|
| `vqa_prompt.py` | The three system prompts (RES, VQA, GCG) with their few-shot examples, the user-prompt template of each of the seven tasks, the style-hint pools, and the rule that assigns a GCG subtask and a style hint to each frame. |
| `generate_vqa_vllm.py` | The generation loop and the validator. Each generated pair must pass every validator rule; a pair that fails is retried once and then discarded. |

## Tasks

| Level | Tasks |
|---|---|
| RES (referring expression segmentation) | `seg_gas`, `seg_source` |
| VQA (scene-understanding question answering) | `ratio_class`, `concentration` |
| GCG (grounded conversation generation) | `gcg_scene`, `gcg_analysis`, `gcg_source` |

Each frame with a plume receives both VQA tasks and one GCG task. The GCG task
is chosen by a 50/30/20 split among `gcg_scene`, `gcg_analysis` and
`gcg_source`, using the MD5 hash of the video and frame name. A frame without
an annotated source that falls in the `gcg_source` bucket receives `gcg_scene`
instead.

The language model receives text only. The prompt carries the gas channel, the
camera model, the CH4 and CO2 percentages, whether the plume and the source are
annotated, the ratio class, and a style hint. It carries no image and no mask.

## Generation settings

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-Next-80B-A3B-Instruct`, bf16 |
| Server | vLLM 0.11.2, OpenAI-compatible API |
| Temperature | 1.0 on the first attempt, 1.1 on the second |
| top_p | 0.95 |
| top_k | 20, from the model's default generation configuration (not set by the client) |
| max_tokens | 512 |
| Seed | MD5 hash of video, frame, level and task, plus 104729 for each retry |
| Attempts | at most 2 per task |
| Result | 212,686 tasks, 212,683 records written, 3 discarded |

## Usage

Start a vLLM server with the model, then run:

```bash
python data_generation/generate_vqa_vllm.py \
    --manifest data/manifest.json \
    --class-csv data/ratio_labels.csv \
    --output data/vqa.jsonl \
    --vllm-url http://localhost:8000/v1 \
    --concurrency 128
```

The script needs the `openai` Python package, listed in
`data_generation/requirements.txt` (`pip install -r data_generation/requirements.txt`). `--manifest` lists the frames
and their metadata, and `--class-csv` gives the ratio class of each video.
