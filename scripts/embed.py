#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable PyTorch device is available")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not mps_backend.is_available():
            raise RuntimeError("MPS was requested, but no MPS-capable PyTorch device is available")

    return requested


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
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Defaults to CUDA, then MPS, then CPU when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.file.read_text(encoding="utf-8")
    device = detect_device(args.device)

    model = AutoModel.from_pretrained(
        args.model_dir.as_posix(),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    with torch.inference_mode():
        embedding = model.encode_text(texts=text, task=args.task)

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().tolist()

    print(json.dumps(embedding))


if __name__ == "__main__":
    main()
