"""Prompt templates for VQA dataset generation via Qwen3-Next-80B.

Unified LLM generation for L1, L2, L3 with metadata-injected required facts.
Every builder returns (system_prompt, user_prompt). Answers use [SEG] (square
brackets), not <SEG>.

Design:
- System prompts carry the FORMAT RULES and DIVERSE FEW-SHOT EXAMPLES so the
  model sees the full range of valid outputs and produces varied phrasings.
- User prompts stay lean: frame metadata + task name + required strings.
- The validator in the generator enforces correctness post-hoc.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

CLASS_BANDS_5 = {
    1: "1-10% CH4, 90-99% CO2",
    2: "11-20% CH4, 80-89% CO2",
    3: "21-30% CH4, 70-79% CO2",
    4: "31-40% CH4, 60-69% CO2",
    5: "41-55% CH4, 45-59% CO2",
}

L3_SUBTASK_WEIGHTS = {
    "gcg_scene": 50,
    "gcg_analysis": 30,
    "gcg_source": 20,
}

L2_TASKS = ("ratio_class", "concentration")
L1_TASKS = ("seg_gas", "seg_source")
L3_SUBTASKS = tuple(L3_SUBTASK_WEIGHTS.keys())


def load_5class_mapping(csv_path: str | Path) -> dict[str, int]:
    """Read ratio_labels.csv and return {video_id: ratio_class}."""
    mapping: dict[str, int] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio_class = int(row["Class"])
            for key in ("Video-CO2", "Video-Methane"):
                vid = row.get(key, "").strip()
                if vid:
                    mapping[vid] = ratio_class
    return mapping


L1_STYLE_HINTS = [
    'Use a direct imperative like "Segment the {target}."',
    'Start the question with "Can you" or "Could you".',
    'Start the question with "Please".',
    'Start the question with "Where is" and end with a segmentation request.',
    'Use "Locate and segment" or "Find and segment" phrasing.',
    'Use "Highlight" or "Show me" instead of "segment".',
    'Use a minimalist answer style like "It is [SEG]." or "[SEG]."',
    'Use an explicit answer style that names the target, like "The {target} is [SEG]."',
]

L2_STYLE_HINTS = [
    'Use "What is" phrasing for the question.',
    'Use "Which" phrasing for the question.',
    'Use "How would you" phrasing (e.g., "How would you classify...").',
    'Use "Identify" or "Determine" as the question verb.',
    'Use "Estimate" or "Quantify" as the question verb.',
    'Use "Give me" or "Tell me" as the question opener.',
    'Use a concise direct style.',
    'Use a slightly more elaborate sentence with a subordinate clause.',
]

L3_STYLE_HINTS = [
    'Start the answer with "A <p>...</p>[SEG]".',
    'Start the answer with "This thermal image shows ...".',
    'Start the answer with "In this frame ...".',
    'Start the answer by naming the camera/modality (gcg_analysis only) or the plume location first.',
    'Use "originating from" to link the plume to its source.',
    'Use "released from" to link the plume to its source.',
    'Use "emitted by" to link the plume to its source.',
    'Use "produced by" to link the plume to its source.',
]


def _pick_style_hint(hints: list[str], frame: dict, task: str) -> str:
    """Deterministically pick one hint per (frame, task) so same task gets
    same hint (no retry thrash) but different frames/tasks get different hints."""
    key = f"{frame['video_id']}/{Path(frame['image_path']).name}/{task}"
    h = hashlib.md5(key.encode()).hexdigest()
    return hints[int(h[:8], 16) % len(hints)]


def assign_l3_subtask(frame: dict) -> str:
    """Deterministic subtask assignment from frame identity hash.

    Falls back to gcg_scene when gcg_source would require an absent source.
    """
    video_id = frame["video_id"]
    frame_name = Path(frame["image_path"]).name
    h = hashlib.md5(f"{video_id}/{frame_name}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100

    cumulative = 0
    chosen = "gcg_scene"
    for subtask, weight in L3_SUBTASK_WEIGHTS.items():
        cumulative += weight
        if bucket < cumulative:
            chosen = subtask
            break

    has_source = frame["mask_stats"].get("has_syringe", False)
    if chosen == "gcg_source" and not has_source:
        return "gcg_scene"
    return chosen


# ---------------------------------------------------------------------------
# Shared frame facts and metadata block
# ---------------------------------------------------------------------------

def _frame_facts(frame: dict) -> dict:
    """Normalize frame metadata used across all builders."""
    stats = frame["mask_stats"]
    cam_modality = frame["camera_type"]  # manifest stores lowercase "co2"/"methane"
    gas_word = "methane" if cam_modality == "methane" else "CO2"
    gas_symbol = "CH4" if cam_modality == "methane" else "CO2"
    modality_display = "methane-sensitive" if cam_modality == "methane" else "CO2-sensitive"
    rc5 = frame["ratio_class_5"]
    ch4 = int(round(float(frame["ch4_pct"])))
    co2 = 100 - ch4  # enforce ch4 + co2 == 100 so validator stays consistent
    # Trim verbose camera model like "FLIR G343 (4.2-4.4µm MWIR)" → "FLIR G343"
    camera_model_raw = frame["camera_model"]
    camera_model_short = camera_model_raw.split(" (")[0].strip()
    return {
        "camera_model": camera_model_short,
        "camera_model_full": camera_model_raw,
        "modality_display": modality_display,
        "gas_word": gas_word,
        "gas_symbol": gas_symbol,
        "ch4_pct": ch4,
        "co2_pct": co2,
        "ratio_class_5": rc5,
        "band": CLASS_BANDS_5[rc5],
        "has_gas": stats["has_gas"],
        "has_source": stats.get("has_syringe", False),
        "quadrant": stats.get("gas_quadrant", "unknown") if stats["has_gas"] else "N/A",
    }


def _frame_metadata_block(frame: dict) -> str:
    f = _frame_facts(frame)
    frame_id = Path(frame["image_path"]).name
    # frame_id is included so same-video frames get distinct prompts and
    # avoid Qwen's prefix-cache determinism.
    # Location info is intentionally OMITTED — the segmentation mask carries
    # location; describing it in text risks hallucination (centroid-based
    # quadrants can disagree with human perception).
    return f"""## Frame Metadata
- Frame ID: {frame_id}
- Camera model: {f['camera_model']}
- Camera type: {f['modality_display']}
- Gas mixture: {f['ch4_pct']}% CH4 / {f['co2_pct']}% CO2
- Ratio class (5-class): Class {f['ratio_class_5']} of 5 ({f['band']})
- Gas plume visible: {f['has_gas']}
- Gas release source visible: {f['has_source']}"""


# ===========================================================================
# LEVEL 1 — Referring Expression Segmentation
# ===========================================================================

SYSTEM_PROMPT_L1 = """\
You generate single-sentence referring expression segmentation QA pairs for a \
thermal Optical Gas Imaging (OGI) VQA dataset. The model answers with a \
special [SEG] token at the end and the segmentation decoder turns it into a \
mask.

# Output format
Return ONE JSON object on a single line, no markdown fences:
{"question": "...", "answer": "... [SEG]"}

# Hard rules
- The answer MUST end with the literal token [SEG] (square brackets).
- The answer is ONE short sentence (LISA-style: minimal, direct).
- The question MUST contain the exact target phrase given by the user (either \
"gas plume" or "gas release source").
- The question may optionally reference the plume location.
- Vary the question phrasing — be natural, direct, and concise.
- NEVER use size adjectives (small, moderate, large, tiny, huge, massive).
- NEVER mention area percentages or coverage percentages.
- NEVER use <p>...</p> tags. Those are for Level 3 only.
- Always call the release point "gas release source" (never "syringe", never brand names).

# Diverse examples (varied phrasings — do NOT copy verbatim)

seg_gas examples:
{"question": "Can you segment the gas plume in this thermal image?", "answer": "Sure, the gas plume is [SEG]."}
{"question": "Please segment the gas plume.", "answer": "It is [SEG]."}
{"question": "Where is the gas plume? Output a segmentation mask.", "answer": "The gas plume is [SEG]."}
{"question": "What is the gas plume in this frame? Please respond with a segmentation mask.", "answer": "Sure, [SEG]."}
{"question": "Show me the gas plume.", "answer": "Here is the gas plume: [SEG]."}
{"question": "Locate and segment the gas plume visible in this thermal image.", "answer": "The gas plume is [SEG]."}
{"question": "Highlight the gas plume in the scene.", "answer": "[SEG]."}
{"question": "Provide a segmentation mask for the gas plume.", "answer": "The gas plume is located at [SEG]."}

seg_source examples:
{"question": "Segment the gas release source.", "answer": "The gas release source is [SEG]."}
{"question": "Can you locate and segment the gas release source?", "answer": "Sure, the gas release source is [SEG]."}
{"question": "Please segment the gas release source in this thermal image.", "answer": "It is [SEG]."}
{"question": "Where is the gas release source? Output a segmentation mask.", "answer": "Here: [SEG]."}
{"question": "Provide a segmentation mask for the gas release source.", "answer": "The gas release source is [SEG]."}
{"question": "Identify and segment the gas release source.", "answer": "Sure, [SEG]."}"""


def build_l1_prompt(frame: dict, task: str) -> tuple[str, str]:
    """L1 referring expression segmentation prompt.

    task in {"seg_gas", "seg_source"}. Returns (system_prompt, user_prompt).
    """
    metadata = _frame_metadata_block(frame)
    f = _frame_facts(frame)

    if task == "seg_gas":
        target = "gas plume"
    elif task == "seg_source":
        target = "gas release source"
    else:
        raise ValueError(f"Unknown L1 task: {task}")

    style = _pick_style_hint(L1_STYLE_HINTS, frame, task).format(target=target)

    user = f"""{metadata}

## Task: {task} (Level 1 Referring Expression Segmentation)

Generate ONE new QA pair asking to segment the "{target}".

Required:
- Question must contain the exact phrase: `{target}`.
- Answer must end with: `[SEG]`.
- Do NOT mention spatial location ("upper-right", "center", etc.).
- Produce a NEW variant — do not repeat the example phrasings verbatim.

Style for THIS call (follow this variation):
- {style}

Output ONE JSON object on a single line:"""
    return SYSTEM_PROMPT_L1, user


# ===========================================================================
# LEVEL 2 — Scene Understanding VQA (text-only)
# ===========================================================================

SYSTEM_PROMPT_L2 = """\
You generate single-sentence VQA pairs about gas composition for a thermal \
Optical Gas Imaging (OGI) dataset. This is a text-only VQA task — no \
segmentation masks, no [SEG] tokens, no <p>...</p> tags.

# Output format
Return ONE JSON object on a single line, no markdown fences:
{"question": "...", "answer": "..."}

# Hard rules
- The answer MUST include every required fact string given by the user.
- The answer is ONE fluent natural-language sentence.
- The question asks about the composition/class; vary phrasing naturally.
- NEVER invent numeric percentages or class numbers beyond those provided.
- NEVER use [SEG] or <p>...</p>.
- NEVER use size adjectives or coverage percentages.

# Diverse examples (varied phrasings — do NOT copy verbatim)

ratio_class examples:
{"question": "Which ratio class does this gas mixture belong to?", "answer": "This gas mixture is Class 2 of 5, corresponding to 11-20% CH4, 80-89% CO2."}
{"question": "How would you classify the gas composition shown here?", "answer": "The composition falls in Class 4 of 5 (31-40% CH4, 60-69% CO2)."}
{"question": "What is the CH4-CO2 ratio class of this frame?", "answer": "It belongs to Class 1 of 5, which is 1-10% CH4 and 90-99% CO2."}
{"question": "Identify the ratio class of the gas mixture in this image.", "answer": "The mixture is classified as Class 3 of 5 (21-30% CH4, 70-79% CO2)."}
{"question": "Assign a class label to the gas composition in this frame.", "answer": "Class 5 of 5 applies here, covering 41-55% CH4 and 45-59% CO2."}

concentration examples:
{"question": "What is the exact gas composition in this image?", "answer": "The mixture contains 15% CH4 and 85% CO2."}
{"question": "Estimate the methane and CO2 concentrations.", "answer": "Approximately 45% methane and 55% carbon dioxide are present in the frame."}
{"question": "Give me the CH4 and CO2 percentages.", "answer": "The gas is 25% CH4 and 75% CO2."}
{"question": "What are the precise methane and carbon dioxide levels?", "answer": "This frame shows 10% methane and 90% CO2."}
{"question": "Quantify the CH4 and CO2 content in this thermal image.", "answer": "The mixture consists of 35% CH4 and 65% CO2."}"""


def build_l2_prompt(frame: dict, task: str) -> tuple[str, str]:
    """L2 scene-understanding VQA prompt.

    task in {"ratio_class", "concentration"}. Returns (system_prompt, user_prompt).
    """
    f = _frame_facts(frame)
    metadata = _frame_metadata_block(frame)

    if task == "ratio_class":
        rc5 = f["ratio_class_5"]
        band = f["band"]
        required = f"""Required (answer must contain, verbatim):
- `Class {rc5} of 5`
- `{band}`  (or both halves separately: `{band.split(', ')[0]}` AND `{band.split(', ')[1]}`)

Forbid: any numeric percentage other than the band edges.
"""
        instruction = "Generate ONE new QA pair asking about the ratio class of the gas mixture."

    elif task == "concentration":
        ch4 = f["ch4_pct"]
        co2 = f["co2_pct"]
        required = f"""Required (answer must contain, verbatim):
- `{ch4}%`  (the CH4 percentage)
- `{co2}%`  (the CO2 percentage)
- One of: `CH4` or `methane`
- One of: `CO2` or `carbon dioxide`

Forbid: any other numeric percentage. ONLY `{ch4}%` and `{co2}%` may appear.
"""
        instruction = "Generate ONE new QA pair asking about the exact CH4/CO2 concentrations."

    else:
        raise ValueError(f"Unknown L2 task: {task}")

    style = _pick_style_hint(L2_STYLE_HINTS, frame, task)

    user = f"""{metadata}

## Task: {task} (Level 2 Scene Understanding VQA)

{instruction}

{required}
Produce a NEW variant — do not repeat one of the system-prompt examples verbatim.

Style for THIS call (follow this variation):
- {style}

Output ONE JSON object on a single line:"""
    return SYSTEM_PROMPT_L2, user


# ===========================================================================
# LEVEL 3 — Grounded Conversation Generation (GLaMM-style)
# ===========================================================================

SYSTEM_PROMPT_L3 = """\
You generate grounded conversation captions for a thermal Optical Gas Imaging \
(OGI) VQA dataset. The model uses interleaved <p>phrase</p>[SEG] tokens to \
ground text spans to segmentation masks (GLaMM convention). The <p>...</p>[SEG] \
tags are MANDATORY STRUCTURE, not optional decoration — every answer MUST use \
them.

# Output format
Return ONE JSON object on a single line, no markdown fences:
{"question": "...", "answer": "..."}

# Hard rules
- Every grounded noun phrase is wrapped in <p>...</p> immediately followed by [SEG].
- Tag ONLY visual entities: "<gas-word> gas plume" and "gas release source". \
Never tag locations, percentages, class numbers, adjectives, or camera names.
- The grounded phrase must read naturally inside the sentence (not appended as \
a trailing tag).
- NEVER use size adjectives (small, moderate, large, tiny, huge).
- NEVER mention area percentages or coverage.
- NEVER invent numeric percentages beyond those provided.
- NEVER describe spatial positions in text ("upper-right", "lower-left", \
"center of the frame", "below it", "above it", "next to", etc.). The \
segmentation mask carries location; the text must NOT speculate about it. \
Describe what is present, not where it is.
- Describe the source only as the release/emission point, without any \
directional modifier ("released from", "emitted by", "originating at").
- Always call the release point "gas release source" (never "syringe", never brand names).

# Three subtasks with diverse examples

## gcg_scene — holistic grounded scene description (no class/percentages/camera)
Write 1–2 fluent sentences describing what is visible. Ground the plume \
(and the source if visible). Do NOT describe spatial locations.

{"question": "Describe what you see in this thermal image with grounded segmentation.", "answer": "A <p>methane gas plume</p>[SEG] is visible, released from a <p>gas release source</p>[SEG]."}
{"question": "Generate a grounded description of this scene.", "answer": "This thermal image shows a <p>CO2 gas plume</p>[SEG] emitted by a <p>gas release source</p>[SEG]."}
{"question": "Caption this OGI frame with grounded phrases.", "answer": "A <p>methane gas plume</p>[SEG] is present in the scene, produced by a <p>gas release source</p>[SEG]."}
{"question": "What does this thermal scene show?", "answer": "The image captures a <p>CO2 gas plume</p>[SEG] originating at a <p>gas release source</p>[SEG]."}

## gcg_analysis — grounded description + composition + camera context
Write 2–3 fluent sentences. Must state the ratio class, exact CH4% and CO2%, \
and reference the camera. Ground the plume (and source if visible). Do NOT \
describe locations.

{"question": "What gas composition is captured in this thermal image?", "answer": "A <p>methane gas plume</p>[SEG] is visible, released from a <p>gas release source</p>[SEG]. Captured with the FLIR Gx320 methane-sensitive camera, the mixture is Class 2 of 5, containing approximately 15% CH4 and 85% CO2."}
{"question": "Analyze the gas mixture and grounding in this frame.", "answer": "This frame shows a <p>CO2 gas plume</p>[SEG] emitted by a <p>gas release source</p>[SEG]. Imaged by the FLIR G343 CO2-sensitive camera, the mixture is Class 1 of 5 with 5% CH4 and 95% CO2."}
{"question": "Describe the gas composition and its origin.", "answer": "The FLIR GF77 methane-sensitive camera captures a <p>methane gas plume</p>[SEG] produced by a <p>gas release source</p>[SEG]. The composition is Class 3 of 5, with 25% CH4 and 75% CO2."}

## gcg_source — grounded causal description (source → plume)
Write 2 fluent sentences. BOTH entities must be grounded. Frame the source as \
the controlled release point and the plume as the resulting emission. Do NOT \
describe spatial positions.

{"question": "Identify the gas release source and the plume it produced.", "answer": "The <p>gas release source</p>[SEG] in this frame is the controlled release point. The resulting <p>methane gas plume</p>[SEG] is the emission it produced."}
{"question": "Describe the causal relationship between source and plume.", "answer": "A <p>gas release source</p>[SEG] emits gas continuously, producing a visible <p>CO2 gas plume</p>[SEG]."}
{"question": "Explain how the visible plume relates to the release point.", "answer": "The <p>gas release source</p>[SEG] is where gas is injected into the scene, and from it a <p>methane gas plume</p>[SEG] develops as the emitted gas."}"""


def build_l3_prompt(frame: dict, subtask: str) -> tuple[str, str]:
    """L3 grounded conversation prompt.

    subtask in {"gcg_scene", "gcg_analysis", "gcg_source"}.
    Returns (system_prompt, user_prompt).
    """
    f = _frame_facts(frame)
    metadata = _frame_metadata_block(frame)

    gas_phrase = f"{f['gas_word']} gas plume"  # "methane gas plume" or "CO2 gas plume"
    gas_tag = f"<p>{gas_phrase}</p>[SEG]"
    source_tag = "<p>gas release source</p>[SEG]"

    header = f"""{metadata}

## Task: {subtask} (Level 3 Grounded Conversation Generation)"""

    if subtask == "gcg_scene":
        if f["has_source"]:
            required_lines = [
                f"- `{gas_tag}`  (grounded plume phrase — mandatory)",
                f"- `{source_tag}`  (grounded source phrase — mandatory)",
            ]
            source_instruction = "Ground BOTH the plume and the source with the tagged phrases."
        else:
            required_lines = [
                f"- `{gas_tag}`  (grounded plume phrase — mandatory)",
            ]
            source_instruction = "Do NOT mention a gas release source — none is visible in this frame."

        body = f"""Generate ONE NEW QA pair: a holistic grounded scene description.

Instructions:
- 1-2 fluent sentences describing what is visible.
- {source_instruction}
- Do NOT describe spatial location (no "upper-right", "center", "below", etc.).
- Do NOT include ratio class number, percentages, or camera model name.
- The <p>...</p>[SEG] tags are MANDATORY.

Required strings (must appear verbatim in answer):
{chr(10).join(required_lines)}

Produce a NEW variant — do not copy a system-prompt example verbatim."""

    elif subtask == "gcg_analysis":
        ch4 = f["ch4_pct"]
        co2 = f["co2_pct"]
        rc5 = f["ratio_class_5"]
        camera = f["camera_model"]
        modality = f["modality_display"]

        source_line = f"- `{source_tag}`  (source is visible — mandatory)" if f["has_source"] else "- (no source phrase — source not visible)"
        source_instruction = (
            "Ground BOTH the plume and the source." if f["has_source"]
            else "Do NOT mention a gas release source — none is visible."
        )

        body = f"""Generate ONE NEW QA pair: a grounded description incorporating gas composition.

Instructions:
- 2-3 fluent sentences.
- {source_instruction}
- State the ratio class explicitly: `Class {rc5} of 5`.
- State the exact percentages: `{ch4}%` CH4 and `{co2}%` CO2.
- Reference the camera naturally, e.g. "captured with the {camera} {modality} camera".
- Do NOT tag percentages, class numbers, or camera names with <p>...</p> — only physical entities.
- Do NOT describe spatial location (no "upper-right", "center", etc.).

Required strings (must appear verbatim in answer):
- `{gas_tag}`  (grounded plume phrase — mandatory)
{source_line}
- `Class {rc5} of 5`
- `{ch4}%`
- `{co2}%`
- `{camera}`

Forbid: any numeric percentage other than `{ch4}%` and `{co2}%`.

Produce a NEW variant — do not copy a system-prompt example verbatim."""

    elif subtask == "gcg_source":
        body = f"""Generate ONE NEW QA pair: a grounded causal description.

Instructions:
- 2 fluent sentences.
- BOTH the plume and the source must be grounded with <p>...</p>[SEG].
- Frame the source as the controlled release point and the plume as the resulting emission.
- Do NOT describe spatial location (no "upper-right", "center", etc.).
- Do NOT include ratio class or percentages.

Required strings (must appear verbatim in answer):
- `{gas_tag}`  (grounded plume phrase — mandatory)
- `{source_tag}`  (grounded source phrase — mandatory)

Produce a NEW variant — do not copy a system-prompt example verbatim."""

    else:
        raise ValueError(f"Unknown L3 subtask: {subtask}")

    style = _pick_style_hint(L3_STYLE_HINTS, frame, subtask)
    user = (
        f"{header}\n\n{body}\n\n"
        f"Style for THIS call (follow this variation):\n- {style}\n\n"
        f"Output ONE JSON object on a single line:"
    )
    return SYSTEM_PROMPT_L3, user
