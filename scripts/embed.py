#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import UTC, datetime
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
TOKEN_CHUNK_LIMIT = 20000


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iso_utc_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat()


def build_record(file_path: Path, modality: str, embedding: list[float], chunk: int) -> dict[str, object]:
    stat = file_path.stat()
    resolved_path = file_path.resolve().as_posix()
    return {
        "status": "ok",
        "record_id": f"{resolved_path}#{chunk}",
        "path": resolved_path,
        "file_name": file_path.name,
        "extension": file_path.suffix.lower(),
        "parent_dir": file_path.parent.resolve().as_posix(),
        "modality": modality,
        "chunk": chunk,
        "size_bytes": stat.st_size,
        "modified_at": iso_utc_timestamp(stat.st_mtime),
        "embedding": embedding,
    }


def chunk_text(tokenizer, text: str, max_tokens: int = TOKEN_CHUNK_LIMIT) -> list[str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_overflowing_tokens=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]

    return [
        chunk
        for chunk in tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if chunk.strip()
    ]


def build_text_records(model, file_path: Path, text: str, task: str) -> list[dict[str, object]]:
    text_chunks = chunk_text(model.processor.tokenizer, text)
    records: list[dict[str, object]] = []

    for chunk_index, chunk_text_value in enumerate(text_chunks):
        with torch.inference_mode():
            embedding = model.encode_text(texts=chunk_text_value, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(build_record(file_path, "text", embedding, chunk_index))

    return records


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
    parser = argparse.ArgumentParser(description="Embed local files or query text using Jina Embeddings V4.")
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
    parser.add_argument("--query-text", help="Query string to embed for semantic search")
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

    if args.query_text is not None:
        with torch.inference_mode():
            embedding = model.encode_text(texts=args.query_text, task=args.task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        print(json.dumps({"status": "query", "embedding": embedding}), flush=True)
        return

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        file_path = Path(raw_line)

        try:
            if is_image_file(file_path):
                with torch.inference_mode():
                    embedding = model.encode_image(images=file_path.as_posix(), task=args.task)

                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().tolist()

                print(json.dumps(build_record(file_path, "image", embedding, 0)), flush=True)
                continue

            text = read_text_file(file_path)
            if text is None:
                print(
                    json.dumps(
                        {
                            "status": "skipped",
                            "path": file_path.resolve().as_posix(),
                            "reason": "unsupported_file_type",
                        }
                    ),
                    flush=True,
                )
                continue

            for record in build_text_records(model, file_path, text, args.task):
                print(json.dumps(record), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "path": file_path.resolve().as_posix(),
                        "reason": str(exc),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
