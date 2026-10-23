#!/usr/bin/env python3
"""Download GasVLM weights from the Hugging Face Hub model repo toqi/GasVLM.

The trained checkpoint is ~2 GB, so it is hosted on the Hub rather than in git.
An ``HF_TOKEN`` environment variable or a cached ``huggingface-cli login`` is
used when present.

Usage:
    python tools/download_weights.py --repo-id toqi/GasVLM --filename gasvlm.pt

The downloaded checkpoint loads into ``GasVLM`` with ``strict=True``.
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True,
                    help="HF model repo id, e.g. toqi/GasVLM")
    ap.add_argument("--filename", default="gasvlm.pt",
                    help="file to fetch from the repo (default: gasvlm.pt)")
    ap.add_argument("--out-dir", default="checkpoints",
                    help="local directory to download into (default: checkpoints/)")
    ap.add_argument("--revision", default=None, help="branch, tag, or commit")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")  # falls back to cached login if None
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=args.out_dir,
        token=token,
    )
    print(f"[download] {args.repo_id}/{args.filename} -> {path}")


if __name__ == "__main__":
    main()
