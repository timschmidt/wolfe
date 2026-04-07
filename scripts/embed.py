#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a local file using Jina Embeddings V4.")
    parser.add_argument("--file", required=True, type=Path, help="Path to the input file")
    parser.add_argument("--model-dir", required=True, type=Path, help="Path to the local model directory")
    parser.add_argument(
        "--task",
        default="retrieval",
        choices=["retrieval", "text-matching", "code"],
        help="Embedding task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.file.read_text(encoding="utf-8")

    model = AutoModel.from_pretrained(
        args.model_dir.as_posix(),
        trust_remote_code=True,
    )
    model.eval()

    with torch.no_grad():
        embedding = model.encode_text(texts=text, task=args.task)

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().tolist()

    print(json.dumps(embedding))


if __name__ == "__main__":
    main()
