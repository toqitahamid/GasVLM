#!/usr/bin/env python3
"""Unpack GasGround WebDataset shards into the directory layout the loader expects.

Produces::

    <out>/{train,val,test}.jsonl
    <out>/{train,val,test}/images/<frame>.png
    <out>/{train,val,test}/masks/<frame>.png

``image_path`` and ``mask_path`` are written to point at ``--out``, so the
directory can be named anything and ``VQADataset`` reads it without changes.

Usage:
    # from an already-downloaded shard directory
    python tools/unpack_gasground.py --tars ./shards --out GasGround

    # or pull the shards from the Hub first (requires `hf auth login`)
    python tools/unpack_gasground.py --repo-id toqi/GasGround --out GasGround
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

SPLITS = ("train", "val", "test")


def fetch(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             allow_patterns=["data/*.tar"])
    return Path(path) / "data"


def unpack_split(tar_dir: Path, out: Path, split: str) -> int:
    shards = sorted(tar_dir.glob(f"{split}-*.tar"))
    if not shards:
        raise FileNotFoundError(f"no shards for split '{split}' in {tar_dir}")
    img_dir = out / split / "images"
    msk_dir = out / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    with open(out / f"{split}.jsonl", "w") as jf:
        for shard in shards:
            with tarfile.open(shard) as tar:
                pending: dict[str, dict] = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    if name.endswith(".mask.png"):
                        key = name[: -len(".mask.png")]
                        (msk_dir / f"{key}.png").write_bytes(tar.extractfile(member).read())
                    elif name.endswith(".png"):
                        key = name[: -len(".png")]
                        (img_dir / f"{key}.png").write_bytes(tar.extractfile(member).read())
                    elif name.endswith(".json"):
                        key = name[: -len(".json")]
                        pending[key] = json.loads(tar.extractfile(member).read())
                    else:
                        continue
                for key, recs in pending.items():
                    for rec in recs:
                        rec["image_path"] = str(out / split / "images" / f"{key}.png")
                        rec["mask_path"] = str(out / split / "masks" / f"{key}.png")
                        jf.write(json.dumps(rec) + "\n")
                    n_frames += 1
    print(f"[unpack] {split}: {n_frames} frames -> {out}")
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tars", type=Path, help="directory holding the .tar shards")
    src.add_argument("--repo-id", type=str, help="Hub dataset repo to download first")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", default=",".join(SPLITS), type=str,
                    help="comma-separated splits to unpack")
    args = ap.parse_args()

    tar_dir = args.tars if args.tars else fetch(args.repo_id)
    args.out.mkdir(parents=True, exist_ok=True)
    splits = tuple(s.strip() for s in args.splits.split(","))
    total = sum(unpack_split(tar_dir, args.out, s) for s in splits)
    print(f"[unpack] {total} frames written to {args.out}")


if __name__ == "__main__":
    main()
