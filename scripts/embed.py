#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


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
    parser = argparse.ArgumentParser(description="Embed local files using Jina Embeddings V4.")
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
    device = detect_device(args.device)

    model = AutoModel.from_pretrained(
        args.model_dir.as_posix(),
        trust_remote_code=True,
    )
    # Jina Embeddings v4 inherits a processor/tokenizer path that needs the Mistral regex fix.
    model.processor = AutoProcessor.from_pretrained(
        args.model_dir.as_posix(),
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=True,
    )
    model.to(device)
    model.eval()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        file_path = Path(raw_line)

        try:
            with torch.inference_mode():
                if is_image_file(file_path):
                    embedding = model.encode_image(images=file_path.as_posix(), task=args.task)
                    modality = "image"
                else:
                    text = read_text_file(file_path)
                    if text is None:
                        continue

                    embedding = model.encode_text(texts=text, task=args.task)
                    modality = "text"

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            response = {
                "path": file_path.as_posix(),
                "modality": modality,
                "embedding": embedding,
            }
            print(json.dumps(response), flush=True)
        except Exception as exc:
            print(f"{file_path}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
