#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

try:
    import lancedb
except ModuleNotFoundError as exc:
    raise SystemExit("lancedb is required; install it with `python -m pip install lancedb`") from exc


BATCH_SIZE = 64
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


def iso_utc_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat()


def build_record(file_path: Path, modality: str, embedding: list[float]) -> dict[str, object]:
    stat = file_path.stat()
    return {
        "path": file_path.resolve().as_posix(),
        "file_name": file_path.name,
        "extension": file_path.suffix.lower(),
        "parent_dir": file_path.parent.resolve().as_posix(),
        "modality": modality,
        "size_bytes": stat.st_size,
        "modified_at": iso_utc_timestamp(stat.st_mtime),
        "embedding": embedding,
    }


def resolve_table_target(db_path: Path) -> tuple[Path, str, Path]:
    if db_path.suffix == ".lance":
        db_root = db_path.parent if db_path.parent != Path("") else Path(".")
        table_name = db_path.stem
        table_path = db_root / f"{table_name}.lance"
        return db_root, table_name, table_path

    return db_path, "embeddings", db_path / "embeddings.lance"


def open_table(db_root: Path, table_name: str):
    db = lancedb.connect(db_root.as_posix())
    try:
        return db.open_table(table_name)
    except Exception:
        return None


def write_records(
    db_root: Path,
    table_name: str,
    table,
    pending_records: list[dict[str, object]],
) -> object:
    if not pending_records:
        return table

    if table is None:
        db = lancedb.connect(db_root.as_posix())
        return db.create_table(table_name, data=pending_records, mode="overwrite")

    merge_insert = getattr(table, "merge_insert", None)
    if callable(merge_insert):
        (
            merge_insert("path")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(pending_records)
        )
    else:
        table.add(pending_records)

    return table


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
    parser.add_argument("--db", default="wolfe.lance", type=Path, help="Path to the Lance table directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db.resolve()
    db_root, table_name, table_path = resolve_table_target(db_path)
    db_root.mkdir(parents=True, exist_ok=True)
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
    table = open_table(db_root, table_name)
    pending_records: list[dict[str, object]] = []
    stored_count = 0
    skipped_count = 0

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
                        skipped_count += 1
                        continue

                    embedding = model.encode_text(texts=text, task=args.task)
                    modality = "text"

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            pending_records.append(build_record(file_path, modality, embedding))
            if len(pending_records) >= BATCH_SIZE:
                table = write_records(db_root, table_name, table, pending_records)
                stored_count += len(pending_records)
                pending_records.clear()
        except Exception as exc:
            print(f"{file_path}: {exc}", file=sys.stderr, flush=True)

    table = write_records(db_root, table_name, table, pending_records)
    stored_count += len(pending_records)
    summary = {
        "db": table_path.as_posix(),
        "table": table_name,
        "stored": stored_count,
        "skipped": skipped_count,
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
