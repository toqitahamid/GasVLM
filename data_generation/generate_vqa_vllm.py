"""Unified VQA generator — L1/L2/L3 via Qwen3-Next-80B-A3B-Instruct on vLLM.

Generates L1 (RES), L2 (VQA), and L3 (GCG) QA pairs with metadata-injected
required facts and strict post-gen validation. Writes a single jsonl at
data/vqa.jsonl, which is then split at the video level into train/val/test.

Usage:
    python data_generation/generate_vqa_vllm.py \
        --manifest data/manifest.json \
        --class-csv data/ratio_labels.csv \
        --output data/vqa.jsonl \
        --vllm-url http://localhost:8000/v1 \
        --concurrency 128
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vqa_prompt import (  # noqa: E402
    L1_TASKS,
    L2_TASKS,
    assign_l3_subtask,
    build_l1_prompt,
    build_l2_prompt,
    build_l3_prompt,
    load_5class_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vqa_gen")


# ---------------------------------------------------------------------------
# Task unit
# ---------------------------------------------------------------------------

@dataclass
class Task:
    frame: dict
    level: int
    task: str  # "seg_gas" | "seg_source" | "ratio_class" | "concentration" |
               # "gcg_scene" | "gcg_analysis" | "gcg_source"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

SIZE_ADJ_PATTERN = re.compile(
    r"\b(tiny|small(?:er)?|moderate(?:ly)?|medium|large|larger|huge|big(?:ger)?|massive)\b",
    re.IGNORECASE,
)
COVERAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%\s*(?:of\s+the\s+(?:image|frame|scene|view|field))",
    re.IGNORECASE,
)
NUMERIC_PCT = re.compile(r"\b(\d{1,3})\s*%")
# Any spatial-location phrase is forbidden everywhere. The mask carries
# location; describing it in text is error-prone and adds no training signal.
SPATIAL_LOCATION_PATTERN = re.compile(
    r"\b("
    r"upper[- ]?(?:left|right)|lower[- ]?(?:left|right)|"
    r"top[- ]?(?:left|right)|bottom[- ]?(?:left|right)|"
    r"center(?:\s+of\s+the\s+(?:image|frame|scene))?|"
    r"middle(?:\s+of\s+the\s+(?:image|frame|scene))?|"
    r"(?:just\s+)?(?:below|above|beneath|underneath|beside|next\s+to|near)\s+(?:it|them)|"
    r"on\s+(?:the\s+)?(?:left|right|top|bottom)(?:\s+side)?|"
    r"to\s+the\s+(?:left|right)"
    r")\b",
    re.IGNORECASE,
)


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _forbidden_generic(text: str) -> Optional[str]:
    if SIZE_ADJ_PATTERN.search(text):
        return "contains size adjective"
    if COVERAGE_PATTERN.search(text):
        return "contains coverage percentage"
    if SPATIAL_LOCATION_PATTERN.search(text):
        return "contains spatial location (forbidden)"
    return None


def _forbidden_numbers(answer: str, allowed: set[int]) -> Optional[str]:
    for m in NUMERIC_PCT.finditer(answer):
        n = int(m.group(1))
        if n not in allowed:
            return f"unexpected numeric {n}%"
    return None


def validate_l1(pair: dict, frame: dict, task: str) -> Optional[str]:
    q = pair.get("question", "")
    a = pair.get("answer", "")
    if not q or not a:
        return "empty q/a"
    if not a.rstrip(" \t\n.!?,").endswith("[SEG]"):
        return "answer does not end with [SEG]"
    target = "gas plume" if task == "seg_gas" else "gas release source"
    if not _contains(q, target):
        return f"question missing target '{target}'"
    if "syringe" in a.lower() or "syringe" in q.lower():
        return "contains forbidden word 'syringe'"
    err = _forbidden_generic(a + " " + q)
    if err:
        return err
    return None


def validate_l2(pair: dict, frame: dict, task: str) -> Optional[str]:
    q = pair.get("question", "")
    a = pair.get("answer", "")
    if not q or not a:
        return "empty q/a"
    if "[SEG]" in a or "<p>" in a:
        return "L2 should not contain [SEG] or <p>"
    err = _forbidden_generic(a)
    if err:
        return err

    ch4 = int(round(float(frame["ch4_pct"])))
    co2 = 100 - ch4
    rc5 = frame["ratio_class_5"]

    if task == "ratio_class":
        if f"class {rc5} of 5" not in a.lower():
            return f"missing 'Class {rc5} of 5'"
        from vqa_prompt import CLASS_BANDS_5
        band = CLASS_BANDS_5[rc5]
        halves = [h.strip() for h in band.split(",")]
        if not any(h.lower() in a.lower() for h in halves):
            return f"missing band substring for Class {rc5}"
    elif task == "concentration":
        if f"{ch4}%" not in a:
            return f"missing '{ch4}%'"
        if f"{co2}%" not in a:
            return f"missing '{co2}%'"
        if not (_contains(a, "CH4") or _contains(a, "methane")):
            return "missing CH4/methane keyword"
        if not (_contains(a, "CO2") or _contains(a, "carbon dioxide")):
            return "missing CO2/carbon dioxide keyword"
        err = _forbidden_numbers(a, allowed={ch4, co2})
        if err:
            return err
    else:
        return f"unknown L2 task {task}"
    return None


def validate_l3(pair: dict, frame: dict, subtask: str) -> Optional[str]:
    q = pair.get("question", "")
    a = pair.get("answer", "")
    if not q or not a:
        return "empty q/a"
    if a.count("<p>") != a.count("</p>"):
        return "unbalanced <p> tags"
    seg_count = a.count("[SEG]")
    p_count = a.count("<p>")
    if seg_count != p_count:
        return f"[SEG]/<p> mismatch ({seg_count} vs {p_count})"
    if p_count == 0:
        return "no grounded phrases"
    # Forbid stray <SEG>
    if "<SEG>" in a:
        return "contains <SEG> (should be [SEG])"

    err = _forbidden_generic(a)
    if err:
        return err

    gas_word = "methane" if frame["camera_type"] == "methane" else "CO2"
    expected_gas_phrase = f"<p>{gas_word} gas plume</p>[SEG]"
    source_phrase_tag = "<p>gas release source</p>[SEG]"

    if expected_gas_phrase not in a:
        return f"missing '{expected_gas_phrase}'"

    has_source = frame["mask_stats"].get("has_syringe", False)
    ch4 = int(round(float(frame["ch4_pct"])))
    co2 = 100 - ch4
    rc5 = frame["ratio_class_5"]
    camera_short = frame["camera_model"].split(" (")[0].strip()

    if subtask == "gcg_scene":
        if has_source and source_phrase_tag not in a:
            return "missing source grounded phrase"
        err = _forbidden_numbers(a, allowed=set())
        if err:
            return err
    elif subtask == "gcg_analysis":
        if has_source and source_phrase_tag not in a:
            return "missing source grounded phrase"
        if f"class {rc5} of 5" not in a.lower():
            return f"missing 'Class {rc5} of 5'"
        if f"{ch4}%" not in a:
            return f"missing '{ch4}%'"
        if f"{co2}%" not in a:
            return f"missing '{co2}%'"
        err = _forbidden_numbers(a, allowed={ch4, co2})
        if err:
            return err
        if camera_short not in a:
            return f"missing camera model '{camera_short}'"
    elif subtask == "gcg_source":
        if source_phrase_tag not in a:
            return "missing source grounded phrase"
        err = _forbidden_numbers(a, allowed=set())
        if err:
            return err
    else:
        return f"unknown L3 subtask {subtask}"
    return None


# ---------------------------------------------------------------------------
# Token positives (for L3)
# ---------------------------------------------------------------------------

def extract_token_positives(answer: str) -> tuple[str, list[list[int]]]:
    phrases = re.findall(r"<p>(.*?)</p>", answer)
    clean = re.sub(r"</?p>|\[SEG\]", "", answer)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not phrases:
        return clean, []
    token_positives: list[list[int]] = []
    search_start = 0
    for phrase in phrases:
        idx = clean.find(phrase, search_start)
        if idx < 0:
            return clean, []
        token_positives.append([idx, idx + len(phrase)])
        search_start = idx + len(phrase)
    return clean, token_positives


def infer_mask_classes(task_name: str, seg_count: int, has_source: bool) -> list[str]:
    if task_name == "seg_gas":
        return ["gas"] * seg_count
    if task_name == "seg_source":
        return ["source"] * seg_count
    # L3 subtasks — gas first, then source if needed
    if seg_count == 1:
        return ["gas"]
    if seg_count == 2:
        return ["gas", "source"] if has_source else ["gas", "gas"]
    return (["gas"] + (["source"] if has_source else []) + ["gas"] * seg_count)[:seg_count]


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------

def create_client(base_url: str):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai SDK not installed. Run: pip install openai")
        sys.exit(1)
    return AsyncOpenAI(base_url=base_url, api_key="EMPTY")


async def call_vllm(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
    top_p: float = 0.95,
) -> Optional[dict]:
    text = ""
    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = await client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse error: %s | raw: %s", e, text[:300])
        return None
    except Exception as e:
        logger.debug("vLLM call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def base_record(frame: dict) -> dict:
    camera_short = frame["camera_model"].split(" (")[0].strip()
    return {
        "video_id": frame["video_id"],
        "image_path": frame["image_path"],
        "mask_path": frame.get("mask_path"),
        "ch4_pct": float(frame["ch4_pct"]),
        "co2_pct": float(frame["co2_pct"]),
        "ratio_class_5": frame["ratio_class_5"],
        "camera_type": frame["camera_type"],
        "camera_model": camera_short,
        "camera_model_full": frame["camera_model"],
    }


def assemble_record(task: Task, pair: dict) -> dict:
    rec = base_record(task.frame)
    rec["level"] = task.level
    rec["task"] = task.task
    rec["question"] = pair["question"].strip()
    answer = pair["answer"].strip()
    rec["answer"] = answer

    if task.level == 3:
        clean, token_positives = extract_token_positives(answer)
        rec["answer_clean"] = clean
        rec["token_positives"] = token_positives
    seg_count = answer.count("[SEG]")
    has_source = task.frame["mask_stats"].get("has_syringe", False)
    if seg_count > 0:
        rec["mask_classes"] = infer_mask_classes(task.task, seg_count, has_source)
    return rec


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def enumerate_tasks(frames: list[dict]) -> list[Task]:
    tasks: list[Task] = []
    for fr in frames:
        stats = fr["mask_stats"]
        has_gas = stats.get("has_gas", False)
        has_source = stats.get("has_syringe", False)

        if has_gas:
            tasks.append(Task(fr, 1, "seg_gas"))
        if has_source:
            tasks.append(Task(fr, 1, "seg_source"))
        if has_gas:
            tasks.append(Task(fr, 2, "ratio_class"))
            tasks.append(Task(fr, 2, "concentration"))
            subtask = assign_l3_subtask(fr)
            tasks.append(Task(fr, 3, subtask))
    return tasks


def build_prompt(task: Task) -> tuple[str, str]:
    if task.level == 1:
        return build_l1_prompt(task.frame, task.task)
    if task.level == 2:
        return build_l2_prompt(task.frame, task.task)
    if task.level == 3:
        return build_l3_prompt(task.frame, task.task)
    raise ValueError(task.level)


def validate(task: Task, pair: dict) -> Optional[str]:
    if task.level == 1:
        return validate_l1(pair, task.frame, task.task)
    if task.level == 2:
        return validate_l2(pair, task.frame, task.task)
    if task.level == 3:
        return validate_l3(pair, task.frame, task.task)
    return "unknown level"


def _task_seed(task: Task) -> int:
    """Deterministic per-task seed — same task always samples the same seed,
    but different tasks (even on same-video frames) get different seeds."""
    key = f"{task.frame['video_id']}/{Path(task.frame['image_path']).name}/{task.level}/{task.task}"
    import hashlib
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


async def process_task(
    client,
    task: Task,
    semaphore: asyncio.Semaphore,
    model: str,
    max_retries: int,
) -> Optional[dict]:
    async with semaphore:
        system_prompt, user_prompt = build_prompt(task)
        # Temperature schedule: high from the start for diversity across
        # same-video frames that share identical metadata.
        temps = [1.0, 1.1, 1.2]
        base_seed = _task_seed(task)
        for attempt in range(max_retries):
            temperature = temps[min(attempt, len(temps) - 1)]
            # Bump seed per attempt so retries explore different samples.
            seed = (base_seed + attempt * 104729) & 0x7FFFFFFF
            pair = await call_vllm(
                client, system_prompt, user_prompt, model,
                temperature=temperature, max_tokens=512,
                seed=seed, top_p=0.95,
            )
            if pair is None:
                continue
            err = validate(task, pair)
            if err is None:
                return assemble_record(task, pair)
            logger.debug(
                "validation fail attempt %d for %s/%s level=%d task=%s: %s | Q=%r A=%r",
                attempt + 1, task.frame["video_id"],
                Path(task.frame["image_path"]).name, task.level, task.task, err,
                pair.get("question", "")[:80], pair.get("answer", "")[:200],
            )
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_frames(manifest_path: Path, class_csv: Path) -> list[dict]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    mapping = load_5class_mapping(class_csv)
    enriched: list[dict] = []
    missing = 0
    for fr in frames:
        vid = fr["video_id"]
        if vid not in mapping:
            missing += 1
            continue
        fr = dict(fr)
        fr["ratio_class_5"] = mapping[vid]
        enriched.append(fr)
    if missing:
        logger.warning("%d frames dropped (video_id not in 5-class csv)", missing)
    logger.info("loaded %d frames with 5-class labels", len(enriched))
    return enriched


def stratified_sample(frames: list[dict], n_total: int, seed: int = 0) -> list[dict]:
    """Sample ~n_total frames stratified by (ratio_class_5, camera_type).

    Picks near-equal counts per bucket, then takes frames from different
    videos within each bucket for variety.
    """
    import random
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = {}
    for fr in frames:
        key = (fr["ratio_class_5"], fr["camera_type"])
        buckets.setdefault(key, []).append(fr)

    n_buckets = len(buckets)
    per_bucket = max(1, n_total // n_buckets)

    selected: list[dict] = []
    for key, items in sorted(buckets.items()):
        by_video: dict[str, list[dict]] = {}
        for it in items:
            by_video.setdefault(it["video_id"], []).append(it)
        videos = sorted(by_video.keys())
        rng.shuffle(videos)
        chosen: list[dict] = []
        vi = 0
        while len(chosen) < per_bucket and videos:
            vid = videos[vi % len(videos)]
            if by_video[vid]:
                chosen.append(by_video[vid].pop(rng.randrange(len(by_video[vid]))))
            else:
                videos.pop(vi % len(videos))
                if not videos:
                    break
                continue
            vi += 1
        selected.extend(chosen)
        logger.info("stratify bucket (class=%d, cam=%s): %d picked from %d videos",
                    key[0], key[1], len(chosen), len(by_video))

    rng.shuffle(selected)
    logger.info("stratified sample: %d frames from %d buckets", len(selected), n_buckets)
    return selected


def task_key(task: Task) -> str:
    return f"{task.frame['image_path']}|L{task.level}|{task.task}"


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def dry_run(args, tasks: list[Task]) -> None:
    print(f"\n=== DRY RUN — {len(tasks)} tasks would be generated ===\n")
    counts: Counter = Counter()
    for i, t in enumerate(tasks):
        counts[(t.level, t.task)] += 1
        system_prompt, user_prompt = build_prompt(t)
        print(f"────────────────────────────────────────────────────────")
        print(f"[{i+1}/{len(tasks)}]  L{t.level}:{t.task}   "
              f"video={t.frame['video_id']}  frame={Path(t.frame['image_path']).name}")
        print(f"  meta: ch4={t.frame['ch4_pct']}% co2={t.frame['co2_pct']}% "
              f"class5={t.frame['ratio_class_5']} cam={t.frame['camera_type']} "
              f"has_source={t.frame['mask_stats'].get('has_syringe')}")
        print(f"  [SYSTEM]\n{system_prompt}\n")
        print(f"  [USER]\n{user_prompt}\n")
    print(f"════════════════════════════════════════════════════════")
    print("Task distribution:")
    for (l, tk), v in sorted(counts.items()):
        print(f"  L{l}:{tk:<15}  {v}")
    print("\n(no vLLM calls made, no output written)")


async def run(args):
    frames = load_frames(Path(args.manifest), Path(args.class_csv))
    if args.stratify:
        frames = stratified_sample(frames, args.stratify, seed=args.stratify_seed)
    elif args.limit:
        frames = frames[: args.limit]

    tasks = enumerate_tasks(frames)
    logger.info("enumerated %d tasks from %d frames", len(tasks), len(frames))

    if args.dry_run:
        dry_run(args, tasks)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress")

    done_keys = load_progress(progress_path)
    pending = [t for t in tasks if task_key(t) not in done_keys]
    logger.info("%d tasks pending (%d already done)", len(pending), len(done_keys))

    if not pending:
        return

    client = create_client(args.vllm_url)
    semaphore = asyncio.Semaphore(args.concurrency)

    written = 0
    failed = 0

    with open(output_path, "a", encoding="utf-8") as out_f, \
         open(progress_path, "a", encoding="utf-8") as prog_f:

        async def handle(task: Task):
            nonlocal written, failed
            rec = await process_task(client, task, semaphore, args.model, args.max_retries)
            if rec is None:
                failed += 1
                return
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            prog_f.write(task_key(task) + "\n")
            prog_f.flush()
            written += 1
            if written % 200 == 0:
                logger.info("written=%d failed=%d", written, failed)

        batch_size = args.concurrency * 4
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            await asyncio.gather(*(handle(t) for t in batch))

    logger.info("DONE. written=%d failed=%d total=%d", written, failed, len(pending))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--class-csv", default="data/ratio_labels.csv")
    p.add_argument("--output", default="data/vqa.jsonl")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stratify", type=int, default=0,
                   help="Sample N frames stratified by (ratio_class_5, camera_type)")
    p.add_argument("--stratify-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts and print them without calling vLLM")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
