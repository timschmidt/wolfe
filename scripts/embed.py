#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fitz
import torch
from PIL import Image
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
PDF_EXTENSIONS = {".pdf"}
TOKEN_CHUNK_LIMIT = 20000
MIN_TOKEN_CHUNK_LIMIT = 1000

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


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


def emit_json(payload: dict[str, object]) -> None:
    try:
        print(json.dumps(payload), flush=True)
    except BrokenPipeError:
        raise SystemExit(0)


def is_cuda_oom(exc: RuntimeError) -> bool:
    return "CUDA out of memory" in str(exc)


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


def encode_text_chunks(model, text: str, task: str, max_tokens: int) -> list[list[float]]:
    text_chunks = chunk_text(model.processor.tokenizer, text, max_tokens=max_tokens)
    embeddings: list[list[float]] = []

    for chunk_text_value in text_chunks:
        try:
            with torch.inference_mode():
                embedding = model.encode_text(
                    texts=chunk_text_value,
                    task=task,
                    max_length=max_tokens,
                    batch_size=1,
                )
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or max_tokens <= MIN_TOKEN_CHUNK_LIMIT:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            embeddings.extend(encode_text_chunks(model, chunk_text_value, task, max_tokens // 2))
            continue

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()
        embeddings.append(embedding)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return embeddings


def build_text_records(model, file_path: Path, text: str, task: str) -> list[dict[str, object]]:
    text_embeddings = encode_text_chunks(model, text, task, TOKEN_CHUNK_LIMIT)
    records: list[dict[str, object]] = []

    for chunk_index, embedding in enumerate(text_embeddings):
        records.append(build_record(file_path, "text", embedding, chunk_index))

    return records


def extract_pdf_content(file_path: Path) -> tuple[str, list[Image.Image]]:
    text_parts: list[str] = []
    page_images: list[Image.Image] = []

    with fitz.open(file_path) as document:
        for page in document:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)

            pixmap = page.get_pixmap(alpha=False)
            page_image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
            page_images.append(page_image)

    return "".join(text_parts), page_images


def build_pdf_records(model, file_path: Path, task: str) -> list[dict[str, object]]:
    pdf_text, page_images = extract_pdf_content(file_path)
    records: list[dict[str, object]] = []

    if pdf_text.strip():
        records.extend(build_text_records(model, file_path, pdf_text, task))

    chunk_offset = len(records)
    for page_index, page_image in enumerate(page_images):
        with torch.inference_mode():
            embedding = model.encode_image(images=page_image, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(build_record(file_path, "image", embedding, chunk_offset + page_index))

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
    if hasattr(model, "verbosity"):
        model.verbosity = 0
    if hasattr(model, "model") and hasattr(model.model, "verbosity"):
        model.model.verbosity = 0

    if args.query_text is not None:
        with torch.inference_mode():
            embedding = model.encode_text(texts=args.query_text, task=args.task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        emit_json({"status": "query", "embedding": embedding})
        return

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        file_path = Path(raw_line)

        try:
            if is_pdf_file(file_path):
                for record in build_pdf_records(model, file_path, args.task):
                    emit_json(record)
                continue

            if is_image_file(file_path):
                with torch.inference_mode():
                    embedding = model.encode_image(images=file_path.as_posix(), task=args.task)

                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().tolist()

                emit_json(build_record(file_path, "image", embedding, 0))
                continue

            text = read_text_file(file_path)
            if text is None:
                emit_json(
                    {
                        "status": "skipped",
                        "path": file_path.resolve().as_posix(),
                        "reason": "unsupported_file_type",
                    }
                )
                continue

            for record in build_text_records(model, file_path, text, args.task):
                emit_json(record)
        except Exception as exc:
            emit_json(
                {
                    "status": "error",
                    "path": file_path.resolve().as_posix(),
                    "reason": str(exc),
                }
            )


if __name__ == "__main__":
    main()
