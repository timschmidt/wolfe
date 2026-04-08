#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fitz
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
TOKEN_CHUNK_LIMIT = 20000
MIN_TOKEN_CHUNK_LIMIT = 1000
YAMNET_SAMPLE_RATE = 16000
YAMNET_TOP_CLASSES = 10
YAMNET_TOP_FRAME_CLASSES = 25
WHISPER_MODEL_ID = "openai/whisper-large-v3"
SPEECH_CLASS_KEYWORDS = (
    "speech",
    "narration",
    "monologue",
    "conversation",
    "whisper",
    "singing",
    "choir",
    "yodel",
    "chant",
    "mantra",
    "rapping",
    "humming",
    "vocal",
    "shout",
    "yell",
    "bellow",
    "whoop",
    "babbling",
)

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

_AUDIO_CLASSIFIER = None
_AUDIO_CLASS_NAMES = None
_WHISPER_PIPE = None


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


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


def load_audio_classifier():
    global _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES
    if _AUDIO_CLASSIFIER is not None:
        return _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES

    classifier = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = []
    with tf.io.gfile.GFile(classifier.class_map_path().numpy()) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    _AUDIO_CLASSIFIER = classifier
    _AUDIO_CLASS_NAMES = class_names
    return _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES


def load_whisper_pipeline(device: str):
    global _WHISPER_PIPE
    if _WHISPER_PIPE is not None:
        return _WHISPER_PIPE

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    pipe_device = device
    if device == "cuda":
        pipe_device = 0
    elif device == "cpu":
        pipe_device = -1

    _WHISPER_PIPE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )
    return _WHISPER_PIPE


def ensure_sample_rate(original_sample_rate: int, waveform: np.ndarray) -> np.ndarray:
    if original_sample_rate != YAMNET_SAMPLE_RATE:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * YAMNET_SAMPLE_RATE))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform.astype(np.float32)


def load_audio_waveform(file_path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(file_path, always_2d=False)
    waveform = np.asarray(waveform)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    else:
        waveform = waveform.astype(np.float32)

    waveform = ensure_sample_rate(sample_rate, waveform)
    return np.clip(waveform, -1.0, 1.0)


def classify_audio_events(file_path: Path) -> tuple[str, list[str]]:
    classifier, class_names = load_audio_classifier()
    waveform = load_audio_waveform(file_path)
    scores, _, _ = classifier(waveform)
    scores_np = scores.numpy()

    mean_scores = scores_np.mean(axis=0)
    top_class_indices = np.argsort(mean_scores)[::-1][:YAMNET_TOP_CLASSES]
    top_class_names = [class_names[index] for index in top_class_indices]

    frame_top_indices = np.argmax(scores_np, axis=1)
    frame_class_names: list[str] = []
    for class_index in frame_top_indices:
        class_name = class_names[class_index]
        if class_name not in frame_class_names:
            frame_class_names.append(class_name)
        if len(frame_class_names) >= YAMNET_TOP_FRAME_CLASSES:
            break

    summary = (
        f"Audio event summary for {file_path.name}. "
        f"Top classes: {', '.join(top_class_names)}. "
        f"Detected frame events: {', '.join(frame_class_names)}."
    )
    return summary, top_class_names + frame_class_names


def should_transcribe_audio(class_names: list[str]) -> bool:
    return any(
        keyword in class_name.lower()
        for class_name in class_names
        for keyword in SPEECH_CLASS_KEYWORDS
    )


def transcribe_audio(file_path: Path, device: str) -> str:
    whisper_pipe = load_whisper_pipeline(device)
    result = whisper_pipe(str(file_path), return_timestamps=False)
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return ""


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


def build_text_records(
    model,
    file_path: Path,
    text: str,
    task: str,
    modality: str = "text",
) -> list[dict[str, object]]:
    text_embeddings = encode_text_chunks(model, text, task, TOKEN_CHUNK_LIMIT)
    records: list[dict[str, object]] = []

    for chunk_index, embedding in enumerate(text_embeddings):
        records.append(build_record(file_path, modality, embedding, chunk_index))

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


def build_audio_records(model, file_path: Path, task: str, device: str) -> list[dict[str, object]]:
    audio_text, class_names = classify_audio_events(file_path)
    records = build_text_records(model, file_path, audio_text, task, modality="audio")

    if should_transcribe_audio(class_names):
        transcript = transcribe_audio(file_path, device)
        if transcript:
            transcript_records = build_text_records(
                model,
                file_path,
                f"Audio transcription for {file_path.name}. {transcript}",
                task,
                modality="audio",
            )
            chunk_offset = len(records)
            for record in transcript_records:
                record["chunk"] = int(record["chunk"]) + chunk_offset
                record["record_id"] = f"{record['path']}#{record['chunk']}"
            records.extend(transcript_records)

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

        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            request = {"type": "file", "path": raw_line}

        request_type = request.get("type", "file")
        if request_type == "shutdown":
            return
        if request_type != "file":
            emit_json({"status": "error", "path": "", "reason": f"unsupported request type: {request_type}"})
            continue

        file_path = Path(request["path"])

        try:
            if is_pdf_file(file_path):
                for record in build_pdf_records(model, file_path, args.task):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_audio_file(file_path):
                for record in build_audio_records(model, file_path, args.task, device):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_image_file(file_path):
                with torch.inference_mode():
                    embedding = model.encode_image(images=file_path.as_posix(), task=args.task)

                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().tolist()

                emit_json(build_record(file_path, "image", embedding, 0))
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
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
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            for record in build_text_records(model, file_path, text, args.task):
                emit_json(record)
            emit_json({"status": "done", "path": file_path.resolve().as_posix()})
        except Exception as exc:
            emit_json(
                {
                    "status": "error",
                    "path": file_path.resolve().as_posix(),
                    "reason": str(exc),
                }
            )
            emit_json({"status": "done", "path": file_path.resolve().as_posix()})


if __name__ == "__main__":
    main()
