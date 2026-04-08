#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor


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
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
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
_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None


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


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def has_video_stream(path: Path) -> bool:
    try:
        streams = ffprobe_streams(path)
    except RuntimeError:
        return False
    return any(stream.get("codec_type") == "video" for stream in streams)


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


def offset_record_chunks(records: list[dict[str, object]], chunk_offset: int) -> list[dict[str, object]]:
    if chunk_offset == 0:
        return records

    for record in records:
        record["chunk"] = int(record["chunk"]) + chunk_offset
        record["record_id"] = f"{record['path']}#{record['chunk']}"
    return records


def require_ffmpeg_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(f"{name} is required for video ingestion but was not found in PATH")
    return binary


def run_process(command: list[str], failure_message: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        if stderr:
            raise RuntimeError(f"{failure_message}: {stderr}") from exc
        raise RuntimeError(failure_message) from exc


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


def load_whisper_model(device: str):
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    if _WHISPER_MODEL is not None and _WHISPER_PROCESSOR is not None:
        return _WHISPER_MODEL, _WHISPER_PROCESSOR

    model_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        dtype=model_dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    _WHISPER_MODEL = model
    _WHISPER_PROCESSOR = processor
    return _WHISPER_MODEL, _WHISPER_PROCESSOR


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


def classify_audio_events(media_path: Path, label_name: str) -> tuple[str, list[str]]:
    classifier, class_names = load_audio_classifier()
    waveform = load_audio_waveform(media_path)
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
        f"Audio event summary for {label_name}. "
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


def transcribe_audio(media_path: Path, device: str) -> str:
    whisper_model, whisper_processor = load_whisper_model(device)
    waveform = load_audio_waveform(media_path)
    inputs = whisper_processor(
        audio=waveform,
        sampling_rate=YAMNET_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs["input_features"].to(device=device, dtype=whisper_model.dtype)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        generated_ids = whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            task="transcribe",
            return_timestamps=False,
        )

    transcript = whisper_processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcript[0].strip() if transcript else ""


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


def build_audio_records(
    model,
    media_path: Path,
    metadata_path: Path,
    task: str,
    device: str,
    label_name: str | None = None,
) -> list[dict[str, object]]:
    media_label = label_name or metadata_path.name
    audio_text, class_names = classify_audio_events(media_path, media_label)
    records = build_text_records(model, metadata_path, audio_text, task, modality="audio")

    if should_transcribe_audio(class_names):
        transcript = transcribe_audio(media_path, device)
        if transcript:
            transcript_records = build_text_records(
                model,
                metadata_path,
                f"Audio transcription for {media_label}. {transcript}",
                task,
                modality="audio",
            )
            records.extend(offset_record_chunks(transcript_records, len(records)))

    return records


def ffprobe_streams(file_path: Path) -> list[dict[str, object]]:
    ffprobe_bin = require_ffmpeg_binary("ffprobe")
    result = run_process(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            file_path.as_posix(),
        ],
        f"ffprobe failed for {file_path.name}",
    )
    payload = json.loads(result.stdout or "{}")
    return payload.get("streams", [])


def extract_video_audio(video_path: Path, output_dir: Path) -> Path | None:
    ffmpeg_bin = require_ffmpeg_binary("ffmpeg")
    audio_path = output_dir / "audio.wav"
    try:
        run_process(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                video_path.as_posix(),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(YAMNET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                audio_path.as_posix(),
            ],
            f"ffmpeg audio extraction failed for {video_path.name}",
        )
    except RuntimeError:
        return None

    return audio_path if audio_path.exists() and audio_path.stat().st_size > 0 else None


def clean_subtitle_text(raw_text: str) -> str:
    cleaned = raw_text.replace("\r\n", "\n")
    cleaned = re.sub(r"\d+\n\d{2}:\d{2}:\d{2}[,.]\d{3} --> .*?(?:\n|$)", "", cleaned)
    cleaned = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> .*?(?:\n|$)", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\{[^}]+\}", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def extract_video_caption_text(video_path: Path, output_dir: Path) -> str:
    streams = ffprobe_streams(video_path)
    subtitle_streams = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    if not subtitle_streams:
        return ""

    ffmpeg_bin = require_ffmpeg_binary("ffmpeg")
    caption_parts: list[str] = []

    for stream in subtitle_streams:
        stream_index = stream.get("index")
        if stream_index is None:
            continue

        subtitle_path = output_dir / f"subtitle-{stream_index}.vtt"
        try:
            run_process(
                [
                    ffmpeg_bin,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    video_path.as_posix(),
                    "-map",
                    f"0:{stream_index}",
                    subtitle_path.as_posix(),
                ],
                f"ffmpeg subtitle extraction failed for {video_path.name}",
            )
        except RuntimeError:
            continue

        if not subtitle_path.exists():
            continue

        subtitle_text = clean_subtitle_text(subtitle_path.read_text(encoding="utf-8", errors="ignore"))
        if subtitle_text:
            caption_parts.append(subtitle_text)

    return "\n".join(caption_parts).strip()


def extract_video_keyframes(video_path: Path, output_dir: Path) -> list[Path]:
    ffmpeg_bin = require_ffmpeg_binary("ffmpeg")
    pattern = output_dir / "frame-%06d.jpg"
    run_process(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-i",
            video_path.as_posix(),
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            "-fps_mode",
            "vfr",
            pattern.as_posix(),
        ],
        f"ffmpeg keyframe extraction failed for {video_path.name}",
    )
    return sorted(output_dir.glob("frame-*.jpg"))


def build_video_records(model, file_path: Path, task: str, device: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="wolfe-video-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        caption_text = extract_video_caption_text(file_path, temp_dir)
        if caption_text:
            caption_records = build_text_records(
                model,
                file_path,
                f"Closed captions for {file_path.name}. {caption_text}",
                task,
                modality="video_text",
            )
            records.extend(offset_record_chunks(caption_records, len(records)))

        audio_path = extract_video_audio(file_path, temp_dir)
        if audio_path is not None:
            audio_records = build_audio_records(
                model,
                audio_path,
                file_path,
                task,
                device,
                label_name=file_path.name,
            )
            if audio_records:
                records.extend(offset_record_chunks(audio_records, len(records)))

        keyframe_paths = extract_video_keyframes(file_path, temp_dir)
        for keyframe_path in keyframe_paths:
            with torch.inference_mode():
                embedding = model.encode_image(images=keyframe_path.as_posix(), task=task)

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            records.append(build_record(file_path, "image", embedding, len(records)))

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
            if is_video_file(file_path) and has_video_stream(file_path):
                for record in build_video_records(model, file_path, args.task, device):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_pdf_file(file_path):
                for record in build_pdf_records(model, file_path, args.task):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_audio_file(file_path):
                for record in build_audio_records(model, file_path, file_path, args.task, device):
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
