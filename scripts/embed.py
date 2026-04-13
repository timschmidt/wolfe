#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import builtins
from enum import Enum
import inspect
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
from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, Qwen2_5OmniForConditionalGeneration


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
QWEN_OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
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
MUSIC_CLASS_KEYWORDS = (
    "music",
    "musical",
    "song",
    "melody",
    "instrument",
    "band",
    "orchestra",
    "symphony",
    "choir",
    "singing",
    "rap",
    "hip hop",
    "dance",
)

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

_AUDIO_CLASSIFIER = None
_AUDIO_CLASS_NAMES = None
_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None
_QWEN_MODEL = None
_QWEN_PROCESSOR = None


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


def normalize_snippet_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_record(
    file_path: Path,
    modality: str,
    embedding: list[float],
    chunk: int,
    offset: int = 0,
    plaintext: str = "",
) -> dict[str, object]:
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
        "offset": offset,
        "plaintext": plaintext,
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


def linear_offsets(count: int, total_units: int) -> list[int]:
    if count <= 0:
        return []
    if total_units <= 0:
        return [0] * count
    return [int(i * total_units / count) for i in range(count)]


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


def unload_whisper_model() -> None:
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    _WHISPER_MODEL = None
    _WHISPER_PROCESSOR = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen_omni_model(device: str):
    global _QWEN_MODEL, _QWEN_PROCESSOR
    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return _QWEN_MODEL, _QWEN_PROCESSOR

    model_dtype = torch.float16 if device == "cuda" else torch.float32
    if "GPTQ" in QWEN_OMNI_MODEL_ID.upper():
        try:
            import auto_gptq  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Qwen GPTQ model selected but auto-gptq is not available. "
                "Install auto-gptq to use the GPTQ checkpoint."
            ) from exc

        quantize_config = getattr(auto_gptq, "QuantizeConfig", None)
        if quantize_config is None:
            quantize_config = getattr(auto_gptq, "BaseQuantizeConfig", None)
        if quantize_config is None:
            raise RuntimeError(
                "auto-gptq is installed but QuantizeConfig is unavailable; "
                "please install a compatible auto-gptq version."
            )
        if quantize_config is not None:
            base_qconfig = quantize_config
            try:
                sig = inspect.signature(base_qconfig)
                allowed_params = set(sig.parameters.keys())
            except (TypeError, ValueError):
                allowed_params = set()

            class _CompatQuantizeConfig(base_qconfig):  # type: ignore[misc]
                def __init__(self, *args, **kwargs):
                    if allowed_params:
                        kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
                    super().__init__(*args, **kwargs)

            builtins.QuantizeConfig = _CompatQuantizeConfig
        if not hasattr(builtins, "METHOD"):
            method = getattr(auto_gptq, "METHOD", None)
            if method is None:
                try:
                    from auto_gptq.utils.import_utils import METHOD as _METHOD  # type: ignore
                except Exception:
                    _METHOD = None
                method = _METHOD
            if method is None:
                class _GPTQMethod(str, Enum):
                    GPTQ = "gptq"
                    EXLLAMA = "exllama"
                    EXLLAMA_V2 = "exllama_v2"
                    TRITON = "triton"

                method = _GPTQMethod
            builtins.METHOD = method
        if not hasattr(builtins, "FORMAT"):
            fmt = getattr(auto_gptq, "FORMAT", None)
            if fmt is None:
                try:
                    from auto_gptq.utils.import_utils import FORMAT as _FORMAT  # type: ignore
                except Exception:
                    _FORMAT = None
                fmt = _FORMAT
            if fmt is None:
                class _GPTQFormat(str, Enum):
                    AUTO = "auto"
                    GPTQ = "gptq"
                    GPTQ_INT4 = "gptq_int4"
                    GPTQ_INT8 = "gptq_int8"

                fmt = _GPTQFormat
            builtins.FORMAT = fmt

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_OMNI_MODEL_ID,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(QWEN_OMNI_MODEL_ID, trust_remote_code=True)

    _QWEN_MODEL = model
    _QWEN_PROCESSOR = processor
    return _QWEN_MODEL, _QWEN_PROCESSOR


def unload_qwen_omni_model() -> None:
    global _QWEN_MODEL, _QWEN_PROCESSOR
    if _QWEN_MODEL is not None:
        try:
            _QWEN_MODEL.to("cpu")
        except Exception:
            pass
    _QWEN_MODEL = None
    _QWEN_PROCESSOR = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class JinaEmbedder:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        model = AutoModel.from_pretrained(
            self.model_dir.as_posix(),
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            self.model_dir.as_posix(),
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        model.processor = processor
        model.to(self.device)
        model.eval()
        if hasattr(model, "verbosity"):
            model.verbosity = 0
        if hasattr(model, "model") and hasattr(model.model, "verbosity"):
            model.model.verbosity = 0
        self.model = model
        self.processor = processor

    def unload(self) -> None:
        if self.model is None and self.processor is None:
            return
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def audio_duration_seconds(file_path: Path) -> int:
    waveform = load_audio_waveform(file_path)
    return int(len(waveform) / YAMNET_SAMPLE_RATE)


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


def should_characterize_music(class_names: list[str]) -> bool:
    return any(
        keyword in class_name.lower()
        for class_name in class_names
        for keyword in MUSIC_CLASS_KEYWORDS
    )


def detect_whisper_language(tokenizer, generated_ids: torch.Tensor) -> str | None:
    if generated_ids is None:
        return None
    lang_to_id = getattr(tokenizer, "lang_to_id", None)
    if not isinstance(lang_to_id, dict):
        return None
    id_to_lang = {value: key for key, value in lang_to_id.items()}
    ids = generated_ids[0].tolist() if generated_ids.ndim > 1 else generated_ids.tolist()
    for token_id in ids:
        if token_id in id_to_lang:
            return id_to_lang[token_id]
    return None


def transcribe_audio(media_path: Path, device: str, language: str | None = None) -> tuple[str, str | None]:
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

    forced_decoder_ids = None
    if language:
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe",
        )

    with torch.inference_mode():
        generated_ids = whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            task="transcribe",
            forced_decoder_ids=forced_decoder_ids,
            return_timestamps=False,
        )

    transcript = whisper_processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    detected_language = language or detect_whisper_language(whisper_processor.tokenizer, generated_ids)
    return (transcript[0].strip() if transcript else ""), detected_language


def describe_music(media_path: Path, device: str) -> str:
    qwen_model, qwen_processor = load_qwen_omni_model(device)
    waveform = load_audio_waveform(media_path)
    prompt = "Please identify genre, instrumentation, mood, and similar works to the attached music."

    inputs = qwen_processor(
        text=prompt,
        audio=waveform,
        sampling_rate=YAMNET_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=256,
        )

    description = qwen_processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return description[0].strip() if description else ""


def chunk_text(tokenizer, text: str, max_tokens: int = TOKEN_CHUNK_LIMIT, start_char: int = 0) -> list[tuple[str, int]]:
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

    decoded_chunks = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    chunks: list[tuple[str, int]] = []
    search_start = 0
    for chunk in decoded_chunks:
        if not chunk.strip():
            continue
        relative_start = text.find(chunk, search_start)
        if relative_start == -1:
            relative_start = search_start
        chunks.append((chunk, start_char + relative_start))
        search_start = relative_start + len(chunk)
    return chunks


def encode_text_chunks(
    embedder: JinaEmbedder,
    text: str,
    task: str,
    max_tokens: int,
    start_char: int = 0,
) -> list[tuple[int, str, list[float]]]:
    embedder.load()
    model = embedder.model
    processor = embedder.processor
    if model is None or processor is None:
        raise RuntimeError("embedding model failed to load")
    text_chunks = chunk_text(processor.tokenizer, text, max_tokens=max_tokens, start_char=start_char)
    embeddings: list[tuple[int, str, list[float]]] = []

    for chunk_text_value, chunk_start in text_chunks:
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
            embeddings.extend(
                encode_text_chunks(
                    embedder,
                    chunk_text_value,
                    task,
                    max_tokens // 2,
                    start_char=chunk_start,
                )
            )
            continue

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()
        embeddings.append((chunk_start, chunk_text_value, embedding))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return embeddings


def build_text_records(
    embedder: JinaEmbedder,
    file_path: Path,
    text: str,
    task: str,
    modality: str = "text",
    offset_fn=None,
    low_memory: bool = False,
    unload_after: bool = True,
) -> list[dict[str, object]]:
    text_embeddings = encode_text_chunks(embedder, text, task, TOKEN_CHUNK_LIMIT)
    records: list[dict[str, object]] = []
    if offset_fn is None:
        offset_fn = lambda char_offset: len(text[:char_offset].encode("utf-8"))

    for chunk_index, (char_offset, chunk_text_value, embedding) in enumerate(text_embeddings):
        records.append(
            build_record(
                file_path,
                modality,
                embedding,
                chunk_index,
                offset_fn(char_offset),
                normalize_snippet_text(chunk_text_value),
            )
        )

    if low_memory and unload_after:
        embedder.unload()
    return records


def extract_pdf_content(file_path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, Image.Image]]]:
    text_parts: list[tuple[int, str]] = []
    page_images: list[tuple[int, Image.Image]] = []

    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text("text")
            if page_text:
                text_parts.append((page_index, page_text))

            pixmap = page.get_pixmap(alpha=False)
            page_image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
            page_images.append((page_index, page_image))

    return text_parts, page_images


def build_pdf_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
    unload_after: bool = True,
) -> list[dict[str, object]]:
    page_texts, page_images = extract_pdf_content(file_path)
    records: list[dict[str, object]] = []

    for page_index, page_text in page_texts:
        if page_text.strip():
            page_records = build_text_records(
                embedder,
                file_path,
                page_text,
                task,
                offset_fn=lambda _char_offset, page_index=page_index: page_index,
                low_memory=low_memory,
                unload_after=False,
            )
            records.extend(offset_record_chunks(page_records, len(records)))

    for page_index, page_image in page_images:
        embedder.load()
        model = embedder.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
        with torch.inference_mode():
            embedding = model.encode_image(images=page_image, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(build_record(file_path, "image", embedding, len(records), page_index))

    if low_memory and unload_after:
        embedder.unload()
    return records


def build_audio_records(
    embedder: JinaEmbedder,
    media_path: Path,
    metadata_path: Path,
    task: str,
    device: str,
    translate: bool,
    music: bool,
    low_memory: bool,
    unload_after: bool = True,
    label_name: str | None = None,
) -> list[dict[str, object]]:
    media_label = label_name or metadata_path.name
    duration_seconds = audio_duration_seconds(media_path)
    audio_text, class_names = classify_audio_events(media_path, media_label)
    records = build_text_records(
        embedder,
        metadata_path,
        audio_text,
        task,
        modality="audio",
        offset_fn=lambda _char_offset: 0,
        low_memory=low_memory,
        unload_after=False,
    )

    if should_transcribe_audio(class_names):
        if low_memory:
            embedder.unload()
        transcript, detected_language = transcribe_audio(media_path, device)
        transcripts: list[tuple[str, str]] = []
        if transcript:
            label = detected_language or "unknown"
            transcripts.append((f"Audio transcription ({label}) for {media_label}. {transcript}", label))

        if translate and detected_language and detected_language != "en":
            translated, _ = transcribe_audio(media_path, device, language="en")
            if translated:
                transcripts.append((f"Audio transcription (en) for {media_label}. {translated}", "en"))

        for transcript_text, _lang in transcripts:
            transcript_records = build_text_records(
                embedder,
                metadata_path,
                transcript_text,
                task,
                modality="audio",
                offset_fn=lambda _char_offset: 0,
                low_memory=low_memory,
                unload_after=False,
            )
            transcript_records = offset_record_chunks(transcript_records, len(records))
            transcript_offsets = linear_offsets(len(transcript_records), duration_seconds)
            for record, offset in zip(transcript_records, transcript_offsets):
                record["offset"] = offset
            records.extend(transcript_records)
        if low_memory and unload_after:
            embedder.unload()
        if low_memory:
            unload_whisper_model()

    if music and should_characterize_music(class_names):
        description = ""
        if low_memory:
            embedder.unload()
            unload_whisper_model()
        try:
            description = describe_music(media_path, device)
        except Exception as exc:
            print(f"music characterization failed for {media_label}: {exc}", file=sys.stderr)
        finally:
            if low_memory:
                unload_qwen_omni_model()
                embedder.load()
        if description:
            description_records = build_text_records(
                embedder,
                metadata_path,
                f"Music characterization for {media_label}. {description}",
                task,
                modality="audio",
                offset_fn=lambda _char_offset: 0,
                low_memory=low_memory,
                unload_after=False,
            )
            description_records = offset_record_chunks(description_records, len(records))
            description_offsets = linear_offsets(len(description_records), duration_seconds)
            for record, offset in zip(description_records, description_offsets):
                record["offset"] = offset
            records.extend(description_records)

    if records:
        record_offsets = linear_offsets(len(records), duration_seconds)
        for record, offset in zip(records, record_offsets):
            record["offset"] = offset

    if low_memory and not unload_after and embedder.model is None:
        embedder.load()

    if low_memory and unload_after:
        embedder.unload()

    return records


def parse_ffprobe_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return float(value)


def get_video_stream_metadata(video_path: Path) -> dict[str, object]:
    streams = ffprobe_streams(video_path)
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"no video stream found in {video_path.name}")
    return video_stream


def time_to_frame_offset(seconds: float, fps: float) -> int:
    if fps <= 0:
        return int(seconds)
    return int(seconds * fps)


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


def parse_subtitle_cues(raw_text: str) -> list[tuple[float, str]]:
    blocks = re.split(r"\n\s*\n", raw_text.replace("\r\n", "\n"))
    cues: list[tuple[float, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in block:
            continue
        timing_line_index = 0
        if "-->" not in lines[0] and len(lines) > 1:
            timing_line_index = 1
        timing_line = lines[timing_line_index]
        match = re.match(
            r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})",
            timing_line,
        )
        if not match:
            continue
        start_text = match.group("start").replace(",", ".")
        hours, minutes, seconds = start_text.split(":")
        start_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        content = clean_subtitle_text("\n".join(lines[timing_line_index + 1 :]))
        if content:
            cues.append((start_seconds, content))
    return cues


def extract_video_caption_text(video_path: Path, output_dir: Path, fps: float) -> list[tuple[int, str]]:
    streams = ffprobe_streams(video_path)
    subtitle_streams = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    if not subtitle_streams:
        return []

    ffmpeg_bin = require_ffmpeg_binary("ffmpeg")
    caption_parts: list[tuple[int, str]] = []

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

        subtitle_text = subtitle_path.read_text(encoding="utf-8", errors="ignore")
        for cue_start_seconds, cue_text in parse_subtitle_cues(subtitle_text):
            caption_parts.append((time_to_frame_offset(cue_start_seconds, fps), cue_text))

    return caption_parts


def extract_video_keyframe_offsets(video_path: Path, fps: float) -> list[int]:
    ffprobe_bin = require_ffmpeg_binary("ffprobe")
    result = run_process(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            video_path.as_posix(),
        ],
        f"ffprobe keyframe scan failed for {video_path.name}",
    )
    payload = json.loads(result.stdout or "{}")
    frames = payload.get("frames", [])
    offsets: list[int] = []
    for frame in frames:
        timestamp = frame.get("best_effort_timestamp_time")
        if timestamp is None:
            continue
        offsets.append(time_to_frame_offset(float(timestamp), fps))
    return offsets


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


def build_video_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    device: str,
    translate: bool,
    music: bool,
    low_memory: bool,
    unload_after: bool = True,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    video_stream = get_video_stream_metadata(file_path)
    fps = parse_ffprobe_rate(
        str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0")
    )

    with tempfile.TemporaryDirectory(prefix="wolfe-video-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        caption_cues = extract_video_caption_text(file_path, temp_dir, fps)
        for cue_offset, cue_text in caption_cues:
            caption_records = build_text_records(
                embedder,
                file_path,
                f"Closed captions for {file_path.name}. {cue_text}",
                task,
                modality="video_text",
                offset_fn=lambda _char_offset, cue_offset=cue_offset: cue_offset,
                low_memory=low_memory,
                unload_after=False,
            )
            records.extend(offset_record_chunks(caption_records, len(records)))

        audio_path = extract_video_audio(file_path, temp_dir)
        if audio_path is not None:
            audio_records = build_audio_records(
                embedder,
                audio_path,
                file_path,
                task,
                device,
                translate,
                music,
                low_memory,
                unload_after=False,
                label_name=file_path.name,
            )
            if audio_records:
                video_duration = float(video_stream.get("duration") or 0.0)
                total_frames = int(video_duration * fps) if video_duration > 0 and fps > 0 else 0
                audio_offsets = linear_offsets(len(audio_records), total_frames)
                for record, offset in zip(audio_records, audio_offsets):
                    record["offset"] = offset
                records.extend(offset_record_chunks(audio_records, len(records)))

        keyframe_offsets = extract_video_keyframe_offsets(file_path, fps)
        keyframe_paths = extract_video_keyframes(file_path, temp_dir)
        for keyframe_index, keyframe_path in enumerate(keyframe_paths):
            embedder.load()
            model = embedder.model
            if model is None:
                raise RuntimeError("embedding model failed to load")
            with torch.inference_mode():
                embedding = model.encode_image(images=keyframe_path.as_posix(), task=task)

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            frame_offset = keyframe_offsets[keyframe_index] if keyframe_index < len(keyframe_offsets) else keyframe_index
            records.append(build_record(file_path, "image", embedding, len(records), frame_offset))

        if low_memory and unload_after:
            embedder.unload()
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
    parser.add_argument(
        "--translate",
        action="store_true",
        help="For non-English audio, run a second Whisper pass forced to English",
    )
    parser.add_argument(
        "--music",
        action="store_true",
        help="Enable music characterization for audio/video when music is detected",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Unload and reload models so only one of Jina or Qwen Omni is in VRAM at a time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    embedder = JinaEmbedder(args.model_dir, device)
    embedder.load()

    if args.query_text is not None:
        embedder.load()
        model = embedder.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
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
                for record in build_video_records(
                    embedder,
                    file_path,
                    args.task,
                    device,
                    args.translate,
                    args.music,
                    args.low_memory,
                    unload_after=False,
                ):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_pdf_file(file_path):
                for record in build_pdf_records(
                    embedder,
                    file_path,
                    args.task,
                    args.low_memory,
                    unload_after=False,
                ):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_audio_file(file_path):
                for record in build_audio_records(
                    embedder,
                    file_path,
                    file_path,
                    args.task,
                    device,
                    args.translate,
                    args.music,
                    args.low_memory,
                    unload_after=False,
                ):
                    emit_json(record)
                emit_json({"status": "done", "path": file_path.resolve().as_posix()})
                continue

            if is_image_file(file_path):
                embedder.load()
                model = embedder.model
                if model is None:
                    raise RuntimeError("embedding model failed to load")
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

            for record in build_text_records(
                embedder,
                file_path,
                text,
                args.task,
                low_memory=args.low_memory,
                unload_after=False,
            ):
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
