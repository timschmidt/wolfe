#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import mailbox
import os
import re
import shutil
import sqlite3
import subprocess
import contextlib
import sys
import tempfile
import zipfile
import tarfile
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path, PurePosixPath
from html import unescape as html_unescape

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7",
)

import fitz
import numpy as np
import scipy.signal
import soundfile as sf
import torch
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    tf = None
    hub = None
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file


IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".psd",
    ".tif",
    ".tiff",
    ".webp",
}
PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".au",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".m4v",
    ".m2ts",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
}
CAD_EXTENSIONS = {
    ".3ds",
    ".gltf",
    ".glb",
    ".obj",
    ".ply",
    ".stl",
    ".vtm",
    ".vti",
    ".vtk",
    ".vtp",
    ".vtr",
    ".vts",
    ".vtu",
    ".wrl",
    ".x3d",
}
FITZ_DOCUMENT_EXTENSIONS = {
    ".djvu",
    ".epub",
    ".ps",
}
SQLITE_EXTENSIONS = {
    ".db3",
    ".sqlite",
    ".sqlite3",
}
NOTEBOOK_EXTENSIONS = {
    ".ipynb",
}
ARCHIVE_EXTENSIONS = {
    ".7z",
    ".cbz",
    ".cbr",
    ".tar",
    ".tbz2",
    ".tgz",
    ".txz",
    ".zip",
}
TOKEN_CHUNK_LIMIT = 20000
MIN_TOKEN_CHUNK_LIMIT = 1000
YAMNET_SAMPLE_RATE = 16000
YAMNET_TOP_CLASSES = 10
YAMNET_TOP_FRAME_CLASSES = 25
JINA_MODEL_ID = "jinaai/jina-embeddings-v4"
WHISPER_MODEL_ID = "openai/whisper-large-v3"
QWEN_OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
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
_TF_GPU_DISABLED = False
_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_DEVICE_MAP_AUTO = False
_ACTIVE_VRAM_MODEL = None


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


def is_cad_file(path: Path) -> bool:
    return path.suffix.lower() in CAD_EXTENSIONS


def is_fitz_document_file(path: Path) -> bool:
    return path.suffix.lower() in FITZ_DOCUMENT_EXTENSIONS


def is_sqlite_file(path: Path) -> bool:
    return path.suffix.lower() in SQLITE_EXTENSIONS


def is_notebook_file(path: Path) -> bool:
    return path.suffix.lower() in NOTEBOOK_EXTENSIONS


def is_archive_file(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes:
        return False
    joined = "".join(suffixes[-2:])
    return path.suffix.lower() in ARCHIVE_EXTENSIONS or joined in {".tar.gz", ".tar.bz2", ".tar.xz"}


# Source of truth is provided by Rust via --document-extensions at startup.
DOCUMENT_EXTENSIONS: set[str] = set()


def is_document_file(path: Path) -> bool:
    return path.suffix.lower() in DOCUMENT_EXTENSIONS


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


def virtualize_records(records: list[dict[str, object]], virtual_path: str) -> list[dict[str, object]]:
    virtual = PurePosixPath(virtual_path)
    parent_dir = virtual.parent.as_posix() if virtual.parent.as_posix() != "." else ""
    extension = virtual.suffix.lower()
    file_name = virtual.name or virtual_path

    for record in records:
        record["path"] = virtual_path
        record["file_name"] = file_name
        record["extension"] = extension
        record["parent_dir"] = parent_dir
        record["record_id"] = f"{virtual_path}#{record['chunk']}"
    return records


def normalize_plaintext(text: str) -> str:
    return normalize_snippet_text(html_unescape(text))


def strip_html_tags(text: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return normalize_plaintext(without_tags)


def build_virtual_path(container_display_path: str, relative_path: str) -> str:
    relative = relative_path.replace("\\", "/").strip("/")
    return f"{container_display_path}!/{relative}" if relative else container_display_path


def sql_identifier(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


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


def require_binary(name: str, feature_name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(f"{name} is required for {feature_name} but was not found in PATH")
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
        stdout = exc.stdout.strip()
        details = stderr or stdout
        if details:
            raise RuntimeError(f"{failure_message}: {details}") from exc
        raise RuntimeError(failure_message) from exc


def load_audio_classifier():
    global _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES
    if _AUDIO_CLASSIFIER is not None:
        return _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES
    if tf is None or hub is None:
        raise RuntimeError("tensorflow and tensorflow_hub are required for audio classification")

    disable_tf_gpu()
    classifier = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = []
    with tf.io.gfile.GFile(classifier.class_map_path().numpy()) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    _AUDIO_CLASSIFIER = classifier
    _AUDIO_CLASS_NAMES = class_names
    return _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES


def unload_audio_classifier() -> None:
    global _AUDIO_CLASSIFIER, _AUDIO_CLASS_NAMES
    _AUDIO_CLASSIFIER = None
    _AUDIO_CLASS_NAMES = None
    if tf is not None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def disable_tf_gpu() -> None:
    """Ensure TensorFlow uses CPU only to save VRAM for other models."""
    global _TF_GPU_DISABLED
    if _TF_GPU_DISABLED:
        return
    if tf is None:
        _TF_GPU_DISABLED = True
        return
    try:
        if tf.config.list_physical_devices("GPU"):
            tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    _TF_GPU_DISABLED = True


def load_whisper_model(device: str):
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    if _WHISPER_MODEL is not None and _WHISPER_PROCESSOR is not None:
        return _WHISPER_MODEL, _WHISPER_PROCESSOR

    model_dtype = torch.float16 if is_cuda_device(device) else torch.float32
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
    global _ACTIVE_VRAM_MODEL
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    if _WHISPER_MODEL is not None:
        try:
            _WHISPER_MODEL.to("cpu")
        except Exception:
            pass
    _WHISPER_MODEL = None
    _WHISPER_PROCESSOR = None
    if _ACTIVE_VRAM_MODEL == "whisper":
        _ACTIVE_VRAM_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen_omni_model(device: str, qwen_max_memory_mb: int | None):
    global _QWEN_MODEL, _QWEN_PROCESSOR
    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return _QWEN_MODEL, _QWEN_PROCESSOR

    if "GPTQ" in QWEN_OMNI_MODEL_ID.upper():
        try:
            config_path = cached_file(QWEN_OMNI_MODEL_ID, "config.json", local_files_only=True)
        except Exception:
            config_path = None
        if config_path:
            try:
                raw_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            except Exception:
                raw_cfg = None
            if isinstance(raw_cfg, dict):
                quant_cfg = raw_cfg.get("quantization_config")
                if isinstance(quant_cfg, dict):
                    updated = False
                    if quant_cfg.get("desc_act") is True:
                        quant_cfg = dict(quant_cfg)
                        quant_cfg["desc_act"] = False
                        if "act_group_aware" in quant_cfg:
                            quant_cfg["act_group_aware"] = False
                        updated = True
                    if "block_name_to_quantize" not in quant_cfg:
                        quant_cfg = dict(quant_cfg)
                        quant_cfg["block_name_to_quantize"] = "thinker.model.layers"
                        updated = True
                    # Force a non-fused kernel for stability with GPTQ weights. This is slower,
                    # but avoids TorchFusedQuantLinear NotImplementedError. Consider revisiting
                    # backend selection for performance once kernels are supported.
                    desired_backend = "torch"
                    if quant_cfg.get("backend") != desired_backend:
                        quant_cfg = dict(quant_cfg)
                        quant_cfg["backend"] = desired_backend
                        quant_cfg["use_exllama"] = False
                        updated = True
                    if updated:
                        raw_cfg["quantization_config"] = quant_cfg
                        Path(config_path).write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")
    config = AutoConfig.from_pretrained(QWEN_OMNI_MODEL_ID, trust_remote_code=True)
    if hasattr(config, "enable_audio_output"):
        config.enable_audio_output = False
    max_memory = None
    device_map = "auto"
    if qwen_max_memory_mb and is_cuda_device(device):
        max_memory = {cuda_device_index(device): f"{qwen_max_memory_mb}MB"}
    with contextlib.redirect_stdout(sys.stderr):
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            QWEN_OMNI_MODEL_ID,
            torch_dtype="auto",
            device_map=device_map,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
            config=config,
            max_memory=max_memory,
        )
        model.eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_OMNI_MODEL_ID, trust_remote_code=True)

    _QWEN_MODEL = model
    _QWEN_PROCESSOR = processor
    global _QWEN_DEVICE_MAP_AUTO
    _QWEN_DEVICE_MAP_AUTO = device_map == "auto"
    return _QWEN_MODEL, _QWEN_PROCESSOR


def unload_qwen_omni_model() -> None:
    global _ACTIVE_VRAM_MODEL
    global _QWEN_MODEL, _QWEN_PROCESSOR
    global _QWEN_DEVICE_MAP_AUTO
    if _QWEN_MODEL is not None and not _QWEN_DEVICE_MAP_AUTO:
        try:
            _QWEN_MODEL.to("cpu")
        except Exception:
            pass
    _QWEN_MODEL = None
    _QWEN_PROCESSOR = None
    _QWEN_DEVICE_MAP_AUTO = False
    if _ACTIVE_VRAM_MODEL == "qwen":
        _ACTIVE_VRAM_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class JinaEmbedder:
    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        model_kwargs = {}
        if is_cuda_device(self.device):
            model_kwargs["dtype"] = cuda_jina_dtype(self.device)
        model = AutoModel.from_pretrained(
            JINA_MODEL_ID,
            trust_remote_code=True,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            JINA_MODEL_ID,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        model.processor = processor
        model.to(self.device)
        patch_jina_batch_autocast(model, self.device)
        model.eval()
        if hasattr(model, "verbosity"):
            model.verbosity = 0
        if hasattr(model, "model") and hasattr(model.model, "verbosity"):
            model.model.verbosity = 0
        self.model = model
        self.processor = processor

    def unload(self) -> None:
        global _ACTIVE_VRAM_MODEL
        if self.model is None and self.processor is None:
            return
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        if _ACTIVE_VRAM_MODEL == "jina":
            _ACTIVE_VRAM_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cuda_jina_dtype(device: str):
    if not is_cuda_device(device):
        return torch.float32
    capability = torch.cuda.get_device_capability(cuda_device_index(device))
    if capability[0] < 8:
        return torch.float32
    return torch.bfloat16


def patch_jina_batch_autocast(model, device: str) -> None:
    if not is_cuda_device(device):
        return

    autocast_dtype = cuda_jina_dtype(device)
    if autocast_dtype == torch.bfloat16:
        return

    def _process_batches(
        self,
        data,
        task_label,
        processor_fn,
        desc,
        return_multivector=False,
        return_numpy=False,
        batch_size=32,
        truncate_dim=None,
    ):
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        dataloader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=processor_fn,
        )
        if return_multivector and len(data) > 1:
            assert not return_numpy, (
                "`return_numpy` is not supported when `return_multivector=True` "
                "and more than one data is encoded"
            )
        results = []
        self.eval()
        for batch in tqdm(dataloader, desc=desc, disable=self.verbosity == 0):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                autocast_context = (
                    contextlib.nullcontext()
                    if autocast_dtype == torch.float32
                    else torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=autocast_dtype,
                    )
                )
                with autocast_context:
                    embeddings = self(**batch, task_label=task_label)
                    if not return_multivector:
                        embeddings = embeddings.single_vec_emb
                        if truncate_dim is not None:
                            embeddings = embeddings[:, :truncate_dim]
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=-1
                            )
                    else:
                        embeddings = embeddings.multi_vec_emb

                    if return_multivector and not return_numpy:
                        valid_tokens = batch["attention_mask"].bool()
                        embeddings = [
                            emb[mask] for emb, mask in zip(embeddings, valid_tokens)
                        ]
                        results.append(embeddings)
                    else:
                        results.append(
                            embeddings.cpu()
                            if return_numpy
                            else list(torch.unbind(embeddings))
                        )
        if return_numpy:
            return np.concatenate([result.numpy() for result in results], axis=0)
        return [item for sublist in results for item in sublist]

    patch_targets = [model]
    if hasattr(model, "get_base_model"):
        try:
            patch_targets.append(model.get_base_model())
        except Exception:
            pass
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        patch_targets.append(model.base_model.model)

    for target in patch_targets:
        if hasattr(target, "_process_batches"):
            target._process_batches = _process_batches.__get__(target, type(target))


def ensure_jina_loaded(embedder: JinaEmbedder, low_memory: bool) -> None:
    global _ACTIVE_VRAM_MODEL
    if low_memory and _ACTIVE_VRAM_MODEL != "jina":
        if _ACTIVE_VRAM_MODEL == "whisper":
            unload_whisper_model()
        elif _ACTIVE_VRAM_MODEL == "qwen":
            unload_qwen_omni_model()
    embedder.load()
    if low_memory:
        _ACTIVE_VRAM_MODEL = "jina"


def ensure_whisper_loaded(embedder: JinaEmbedder, device: str, low_memory: bool):
    global _ACTIVE_VRAM_MODEL
    if low_memory and _ACTIVE_VRAM_MODEL != "whisper":
        if _ACTIVE_VRAM_MODEL == "jina":
            embedder.unload()
        elif _ACTIVE_VRAM_MODEL == "qwen":
            unload_qwen_omni_model()
    model, processor = load_whisper_model(device)
    if low_memory:
        _ACTIVE_VRAM_MODEL = "whisper"
    return model, processor


def ensure_qwen_loaded(embedder: JinaEmbedder, device: str, qwen_max_memory_mb: int | None, low_memory: bool):
    global _ACTIVE_VRAM_MODEL
    if low_memory and _ACTIVE_VRAM_MODEL != "qwen":
        if _ACTIVE_VRAM_MODEL == "jina":
            embedder.unload()
        elif _ACTIVE_VRAM_MODEL == "whisper":
            unload_whisper_model()
    model, processor = load_qwen_omni_model(device, qwen_max_memory_mb)
    if low_memory:
        _ACTIVE_VRAM_MODEL = "qwen"
    return model, processor


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


def classify_audio_events(media_path: Path, label_name: str, low_memory: bool = False) -> tuple[str, list[str]]:
    if low_memory:
        disable_tf_gpu()
    classifier, class_names = load_audio_classifier()
    waveform = load_audio_waveform(media_path)
    with tf.device("/CPU:0"):
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
    if low_memory:
        unload_audio_classifier()
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


def transcribe_audio_with_model(
    media_path: Path,
    device: str,
    whisper_model,
    whisper_processor,
    language: str | None = None,
) -> tuple[str, str | None]:
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


def describe_music_with_model(media_path: Path, device: str, qwen_model, qwen_processor) -> str:
    waveform = load_audio_waveform(media_path)
    prompt = (
        "Fill out this profile about the music you hear. Be thorough.\n\n"
        'Instrumentation/Vocals: ""\n'
        'Soundscape: ""\n'
        'Mood: ""\n'
        'Genre: ""\n'
        'Style: ""\n'
        'Description: ""\n'
        'Comment: ""\n'
        'Progression: ""\n'
        'Similar works: ""\n\n'
        "(Progression means how the song evolves or if there are notable changes or moments.)"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": waveform},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = qwen_processor(
        text=text,
        audio=waveform,
        sampling_rate=YAMNET_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}

    with contextlib.redirect_stdout(sys.stderr):
        with torch.inference_mode():
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=2048,
                use_cache=False,
            )

    input_ids = inputs.get("input_ids")
    if input_ids is not None and hasattr(input_ids, "shape"):
        prompt_length = input_ids.shape[-1]
        if generated_ids.shape[-1] > prompt_length:
            generated_ids = generated_ids[:, prompt_length:]

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
    low_memory: bool = False,
) -> list[tuple[int, str, list[float]]]:
    ensure_jina_loaded(embedder, low_memory)
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
                    low_memory=low_memory,
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
) -> list[dict[str, object]]:
    text_embeddings = encode_text_chunks(embedder, text, task, TOKEN_CHUNK_LIMIT, low_memory=low_memory)
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
            )
            records.extend(offset_record_chunks(page_records, len(records)))

    for page_index, page_image in page_images:
        ensure_jina_loaded(embedder, low_memory)
        model = embedder.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
        with torch.inference_mode():
            embedding = model.encode_image(images=page_image, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(build_record(file_path, "image", embedding, len(records), page_index))

    return records


def build_pdf_records_from_source(
    embedder: JinaEmbedder,
    source_path: Path,
    pdf_path: Path,
    task: str,
    low_memory: bool,
) -> list[dict[str, object]]:
    page_texts, page_images = extract_pdf_content(pdf_path)
    records: list[dict[str, object]] = []

    for page_index, page_text in page_texts:
        if page_text.strip():
            page_records = build_text_records(
                embedder,
                source_path,
                page_text,
                task,
                offset_fn=lambda _char_offset, page_index=page_index: page_index,
                low_memory=low_memory,
            )
            records.extend(offset_record_chunks(page_records, len(records)))

    for page_index, page_image in page_images:
        ensure_jina_loaded(embedder, low_memory)
        model = embedder.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
        with torch.inference_mode():
            embedding = model.encode_image(images=page_image, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(
            build_record(
                source_path,
                "image",
                embedding,
                len(records),
                page_index,
            )
        )

    return records


def strip_ansi_escape_codes(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_f3d_metadata_output(output: str, file_path: Path) -> str:
    cleaned_lines = [line.strip() for line in strip_ansi_escape_codes(output).splitlines()]
    metadata_lines: list[str] = []

    for line in cleaned_lines:
        if not line:
            continue
        if line.startswith("Found a reader for"):
            metadata_lines.append(line)
            continue
        if line.startswith("Number of files:"):
            metadata_lines.append(line)
            continue
        if line.startswith("Number of actors:"):
            metadata_lines.append(line)
            continue
        if line.startswith("Scene bounding box:"):
            metadata_lines.append(line)
            continue
        if line.startswith("Camera position:"):
            metadata_lines.append(line)
            continue
        if line.startswith("Camera focal point:"):
            metadata_lines.append(line)
            continue
        if line.startswith("Camera view up:"):
            metadata_lines.append(line)
            continue
        if line.startswith("data output "):
            metadata_lines.append(line)
            continue
        if line.startswith("Version:") or line.startswith("Description:"):
            metadata_lines.append(line)

    prefix = f"CAD metadata for {file_path.name}. Format: {file_path.suffix.lower()}."
    if metadata_lines:
        return f"{prefix} {' '.join(metadata_lines)}"
    return prefix


def extract_f3d_metadata(file_path: Path) -> str:
    f3d_bin = require_binary("f3d", "CAD ingestion")
    result = run_process(
        [
            f3d_bin,
            "--no-config",
            "--no-render",
            "--verbose=debug",
            file_path.as_posix(),
        ],
        f"f3d metadata extraction failed for {file_path.name}",
    )
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return parse_f3d_metadata_output(combined_output, file_path)


def render_f3d_view(file_path: Path, output_path: Path, direction: str, view_up: str) -> None:
    f3d_bin = require_binary("f3d", "CAD ingestion")
    try:
        run_process(
            [
                f3d_bin,
                "--no-config",
                "--output",
                output_path.as_posix(),
                "--resolution",
                "1024,1024",
                "--camera-direction",
                direction,
                "--camera-view-up",
                view_up,
                "--camera-orthographic",
                "--axis=0",
                "--grid=0",
                "--filename=0",
                "--metadata=0",
                "--loading-progress=0",
                "--animation-progress=0",
                "--background-color",
                "#ffffff",
                "--edges=1",
                file_path.as_posix(),
            ],
            f"f3d render failed for {file_path.name}",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"f3d render failed for {file_path.name}: {exc}. Ensure F3D has a usable rendering backend for offscreen output."
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"f3d render did not produce an image for {file_path.name}. Ensure F3D can render offscreen on this machine."
        )


def build_cad_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    metadata_text = extract_f3d_metadata(file_path)
    if metadata_text.strip():
        metadata_records = build_text_records(
            embedder,
            file_path,
            metadata_text,
            task,
            modality="cad_text",
            offset_fn=lambda _char_offset: 0,
            low_memory=low_memory,
        )
        records.extend(offset_record_chunks(metadata_records, len(records)))

    views = (
        ("front", "-Z", "+Y"),
        ("back", "+Z", "+Y"),
        ("left", "+X", "+Y"),
        ("right", "-X", "+Y"),
        ("top", "-Y", "+Z"),
        ("bottom", "+Y", "-Z"),
        ("isometric", "-1,-1,-1", "+Y"),
    )

    with tempfile.TemporaryDirectory(prefix="wolfe-cad-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for view_index, (view_name, direction, view_up) in enumerate(views):
            image_path = temp_dir / f"{view_index:02d}-{view_name}.png"
            render_f3d_view(file_path, image_path, direction, view_up)
            ensure_jina_loaded(embedder, low_memory)
            model = embedder.model
            if model is None:
                raise RuntimeError("embedding model failed to load")
            with torch.inference_mode():
                embedding = model.encode_image(images=image_path.as_posix(), task=task)

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            records.append(
                build_record(
                    file_path,
                    "image",
                    embedding,
                    len(records),
                    view_index,
                    plaintext=f"cad view: {view_name}",
                )
            )

    return records


def build_image_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
) -> list[dict[str, object]]:
    ensure_jina_loaded(embedder, low_memory)
    model = embedder.model
    if model is None:
        raise RuntimeError("embedding model failed to load")
    with torch.inference_mode():
        embedding = model.encode_image(images=file_path.as_posix(), task=task)

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().tolist()

    return [build_record(file_path, "image", embedding, 0)]


def extract_fitz_document_content(file_path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, Image.Image]]]:
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


def build_fitz_document_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
) -> list[dict[str, object]]:
    page_texts, page_images = extract_fitz_document_content(file_path)
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
            )
            records.extend(offset_record_chunks(page_records, len(records)))

    for page_index, page_image in page_images:
        ensure_jina_loaded(embedder, low_memory)
        model = embedder.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
        with torch.inference_mode():
            embedding = model.encode_image(images=page_image, task=task)

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()

        records.append(build_record(file_path, "image", embedding, len(records), page_index))

    return records


def normalize_email_part_text(content_type: str, payload: str) -> str:
    if content_type == "text/html":
        return strip_html_tags(payload)
    return normalize_plaintext(payload)


def extract_email_body_text(message) -> str:
    parts: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            disposition = (part.get_content_disposition() or "").lower()
            if disposition == "attachment":
                continue
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            try:
                payload = part.get_content()
            except Exception:
                continue
            if isinstance(payload, str):
                normalized = normalize_email_part_text(content_type, payload)
                if normalized:
                    parts.append(normalized)
    else:
        try:
            payload = message.get_content()
        except Exception:
            payload = ""
        if isinstance(payload, str):
            normalized = normalize_email_part_text(message.get_content_type(), payload)
            if normalized:
                parts.append(normalized)
    return "\n\n".join(parts)


def build_email_header_text(message, label: str) -> str:
    fields = []
    for header in ("subject", "from", "to", "cc", "bcc", "date", "message-id"):
        value = message.get(header)
        if value:
            fields.append(f"{header.title()}: {normalize_plaintext(str(value))}")
    body = extract_email_body_text(message)
    combined = [f"Email metadata for {label}."] + fields
    if body:
        combined.append(f"Body: {body}")
    return "\n".join(combined)


def iter_email_attachments(message):
    for part in message.iter_attachments():
        filename = part.get_filename()
        if not filename:
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            payload = None
        if not payload:
            continue
        yield filename, payload


def build_eml_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    device: str,
    translate: bool,
    low_memory: bool,
    qwen_max_memory_mb: int | None,
    display_path: str,
    depth: int,
) -> list[dict[str, object]]:
    with file_path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    records = build_text_records(
        embedder,
        file_path,
        build_email_header_text(message, Path(display_path).name),
        task,
        low_memory=low_memory,
    )
    records = virtualize_records(records, display_path)

    with tempfile.TemporaryDirectory(prefix="wolfe-eml-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for attachment_index, (filename, payload) in enumerate(iter_email_attachments(message)):
            attachment_path = temp_dir / f"{attachment_index:03d}-{Path(filename).name}"
            attachment_path.write_bytes(payload)
            attachment_display = build_virtual_path(display_path, filename)
            attachment_records = build_records_for_file(
                embedder,
                attachment_path,
                task,
                device,
                translate,
                low_memory,
                qwen_max_memory_mb,
                display_path=attachment_display,
                depth=depth + 1,
            )
            records.extend(offset_record_chunks(attachment_records, len(records)))

    return records


def build_mbox_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
    display_path: str,
) -> list[dict[str, object]]:
    box = mailbox.mbox(file_path.as_posix())
    message_texts: list[str] = []
    try:
        for message_index, message in enumerate(box):
            label = f"{Path(display_path).name} message {message_index + 1}"
            message_text = build_email_header_text(message, label)
            if message_text.strip():
                message_texts.append(message_text)
    finally:
        box.close()

    records: list[dict[str, object]] = []
    for message_index, message_text in enumerate(message_texts):
        message_records = build_text_records(
            embedder,
            file_path,
            message_text,
            task,
            offset_fn=lambda _char_offset, message_index=message_index: message_index,
            low_memory=low_memory,
        )
        records.extend(offset_record_chunks(message_records, len(records)))

    return virtualize_records(records, display_path)


def build_sqlite_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
    display_path: str,
) -> list[dict[str, object]]:
    connection = sqlite3.connect(f"file:{file_path.as_posix()}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        tables = cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        parts = [f"SQLite metadata for {Path(display_path).name}. Table count: {len(tables)}."]
        for table_name, schema_sql in tables[:50]:
            parts.append(f"Table {table_name}.")
            if schema_sql:
                parts.append(f"Schema: {normalize_plaintext(schema_sql)}")
            columns = cursor.execute(f"PRAGMA table_info({sql_identifier(table_name)})").fetchall()
            if columns:
                column_text = ", ".join(f"{column[1]} {column[2]}" for column in columns)
                parts.append(f"Columns: {column_text}.")
            sample_rows = cursor.execute(f"SELECT * FROM {sql_identifier(table_name)} LIMIT 3").fetchall()
            if sample_rows:
                serialized_rows = []
                for row in sample_rows:
                    serialized_rows.append(normalize_plaintext(json.dumps(row, default=str)))
                parts.append(f"Sample rows: {'; '.join(serialized_rows)}.")
        text = "\n".join(parts)
    finally:
        connection.close()

    records = build_text_records(
        embedder,
        file_path,
        text,
        task,
        low_memory=low_memory,
    )
    return virtualize_records(records, display_path)


def stringify_notebook_output(output: dict[str, object]) -> str:
    text_chunks: list[str] = []
    for key in ("text", "text/plain", "application/json"):
        value = output.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            text_chunks.append("".join(str(item) for item in value))
        else:
            text_chunks.append(str(value))
    data = output.get("data")
    if isinstance(data, dict):
        for key in ("text/plain", "application/json", "text/html"):
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                text_chunks.append("".join(str(item) for item in value))
            else:
                text_chunks.append(str(value))
    return normalize_plaintext("\n".join(text_chunks))


def build_notebook_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    low_memory: bool,
    display_path: str,
) -> list[dict[str, object]]:
    notebook = json.loads(file_path.read_text(encoding="utf-8"))
    parts = [f"Notebook content for {Path(display_path).name}."]
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        normalized_source = normalize_plaintext(source_text)
        if normalized_source:
            parts.append(f"{cell_type.title()} cell {cell_index + 1}: {normalized_source}")
        outputs = cell.get("outputs", [])
        for output in outputs:
            if not isinstance(output, dict):
                continue
            output_text = stringify_notebook_output(output)
            if output_text:
                parts.append(f"Output for cell {cell_index + 1}: {output_text}")

    records = build_text_records(
        embedder,
        file_path,
        "\n".join(parts),
        task,
        low_memory=low_memory,
    )
    return virtualize_records(records, display_path)


def extract_archive_to_dir(file_path: Path, output_dir: Path) -> None:
    name = file_path.name.lower()
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as archive:
            archive.extractall(output_dir)
        return
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as archive:
            archive.extractall(output_dir)
        return
    if name.endswith(".7z"):
        seven_zip = require_binary("7z", "archive ingestion")
        run_process([seven_zip, "x", f"-o{output_dir.as_posix()}", file_path.as_posix()], f"7z extraction failed for {file_path.name}")
        return
    if name.endswith(".cbr"):
        if shutil.which("unrar"):
            run_process(
                [require_binary("unrar", "archive ingestion"), "x", "-idq", file_path.as_posix(), output_dir.as_posix()],
                f"unrar extraction failed for {file_path.name}",
            )
            return
        seven_zip = require_binary("7z", "archive ingestion")
        run_process([seven_zip, "x", f"-o{output_dir.as_posix()}", file_path.as_posix()], f"7z extraction failed for {file_path.name}")
        return
    raise RuntimeError(f"unsupported archive format for {file_path.name}")


def iter_extracted_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_archive_summary_text(file_path: Path, extracted_files: list[Path], archive_display_path: str, extraction_root: Path) -> str:
    relative_names = [path.relative_to(extraction_root).as_posix() for path in extracted_files[:200]]
    parts = [
        f"Archive contents for {Path(archive_display_path).name}.",
        f"Contained file count: {len(extracted_files)}.",
    ]
    if relative_names:
        parts.append(f"Contained files: {', '.join(relative_names)}.")
    return " ".join(parts)

def convert_document_to_pdf(path: Path) -> tuple[Path | None, tempfile.TemporaryDirectory | None, str | None]:
    soffice_path = shutil.which("soffice")
    if not soffice_path:
        return None, None, "libreoffice_missing"

    tmpdir = tempfile.TemporaryDirectory()
    outdir = Path(tmpdir.name)
    try:
        result = subprocess.run(
            [
                soffice_path,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                outdir.as_posix(),
                path.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        tmpdir.cleanup()
        return None, None, "libreoffice_conversion_failed"

    expected_name = path.with_suffix(".pdf").name
    pdf_path = outdir / expected_name
    if not pdf_path.exists():
        pdf_candidates = list(outdir.glob("*.pdf"))
        if pdf_candidates:
            pdf_path = pdf_candidates[0]

    if not pdf_path.exists():
        tmpdir.cleanup()
        return None, None, "libreoffice_conversion_failed"

    return pdf_path, tmpdir, None


def build_audio_records(
    embedder: JinaEmbedder,
    media_path: Path,
    metadata_path: Path,
    task: str,
    device: str,
    translate: bool,
    low_memory: bool,
    qwen_max_memory_mb: int | None,
    label_name: str | None = None,
) -> list[dict[str, object]]:
    media_label = label_name or metadata_path.name
    duration_seconds = audio_duration_seconds(media_path)
    audio_text, class_names = classify_audio_events(media_path, media_label, low_memory=low_memory)
    records = build_text_records(
        embedder,
        metadata_path,
        audio_text,
        task,
        modality="audio",
        offset_fn=lambda _char_offset: 0,
        low_memory=low_memory,
    )

    if should_transcribe_audio(class_names):
        whisper_model, whisper_processor = ensure_whisper_loaded(embedder, device, low_memory)
        transcript, detected_language = transcribe_audio_with_model(
            media_path,
            device,
            whisper_model,
            whisper_processor,
        )
        transcripts: list[tuple[str, str]] = []
        if transcript:
            label = detected_language or "unknown"
            transcripts.append((f"Audio transcription ({label}) for {media_label}. {transcript}", label))

        if translate and detected_language and detected_language != "en":
            translated, _ = transcribe_audio_with_model(
                media_path,
                device,
                whisper_model,
                whisper_processor,
                language="en",
            )
            if translated:
                transcripts.append((f"Audio transcription (en) for {media_label}. {translated}", "en"))

        if low_memory:
            unload_whisper_model()
            del whisper_model, whisper_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for transcript_text, _lang in transcripts:
            transcript_records = build_text_records(
                embedder,
                metadata_path,
                transcript_text,
                task,
                modality="audio",
                offset_fn=lambda _char_offset: 0,
                low_memory=low_memory,
            )
            transcript_records = offset_record_chunks(transcript_records, len(records))
            transcript_offsets = linear_offsets(len(transcript_records), duration_seconds)
            for record, offset in zip(transcript_records, transcript_offsets):
                record["offset"] = offset
            records.extend(transcript_records)

    if should_characterize_music(class_names):
        description = ""
        try:
            qwen_model, qwen_processor = ensure_qwen_loaded(embedder, device, qwen_max_memory_mb, low_memory)
            description = describe_music_with_model(media_path, device, qwen_model, qwen_processor)
        except Exception as exc:
            print(f"music characterization failed for {media_label}: {exc}", file=sys.stderr)
        if low_memory:
            unload_qwen_omni_model()
            del qwen_model, qwen_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if description:
            description_records = build_text_records(
                embedder,
                metadata_path,
                f"Music characterization for {media_label}. {description}",
                task,
                modality="audio",
                offset_fn=lambda _char_offset: 0,
                low_memory=low_memory,
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
    low_memory: bool,
    qwen_max_memory_mb: int | None,
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
                low_memory,
                qwen_max_memory_mb,
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
            ensure_jina_loaded(embedder, low_memory)
            model = embedder.model
            if model is None:
                raise RuntimeError("embedding model failed to load")
            with torch.inference_mode():
                embedding = model.encode_image(images=keyframe_path.as_posix(), task=task)

            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().tolist()

            frame_offset = keyframe_offsets[keyframe_index] if keyframe_index < len(keyframe_offsets) else keyframe_index
            records.append(build_record(file_path, "image", embedding, len(records), frame_offset))

    return records


def build_archive_records(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    device: str,
    translate: bool,
    low_memory: bool,
    qwen_max_memory_mb: int | None,
    display_path: str,
    depth: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="wolfe-archive-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extract_archive_to_dir(file_path, temp_dir)
        extracted_files = iter_extracted_files(temp_dir)

        summary_records = build_text_records(
            embedder,
            file_path,
            build_archive_summary_text(file_path, extracted_files, display_path, temp_dir),
            task,
            modality="archive_text",
            offset_fn=lambda _char_offset: 0,
            low_memory=low_memory,
        )
        summary_records = virtualize_records(summary_records, display_path)
        records.extend(offset_record_chunks(summary_records, len(records)))

        for extracted_file in extracted_files:
            relative_path = extracted_file.relative_to(temp_dir).as_posix()
            child_display = build_virtual_path(display_path, relative_path)
            child_records = build_records_for_file(
                embedder,
                extracted_file,
                task,
                device,
                translate,
                low_memory,
                qwen_max_memory_mb,
                display_path=child_display,
                depth=depth + 1,
            )
            records.extend(offset_record_chunks(child_records, len(records)))
    return records


def build_records_for_file(
    embedder: JinaEmbedder,
    file_path: Path,
    task: str,
    device: str,
    translate: bool,
    low_memory: bool,
    qwen_max_memory_mb: int | None,
    display_path: str | None = None,
    depth: int = 0,
) -> list[dict[str, object]]:
    if depth > 4:
        raise RuntimeError(f"nested container depth exceeded for {file_path.name}")

    resolved_display_path = display_path or file_path.resolve().as_posix()
    records: list[dict[str, object]]
    already_virtualized = False

    if is_video_file(file_path) and has_video_stream(file_path):
        records = build_video_records(
            embedder,
            file_path,
            task,
            device,
            translate,
            low_memory,
            qwen_max_memory_mb,
        )
    elif is_pdf_file(file_path):
        records = build_pdf_records(
            embedder,
            file_path,
            task,
            low_memory,
        )
    elif is_fitz_document_file(file_path):
        records = build_fitz_document_records(
            embedder,
            file_path,
            task,
            low_memory,
        )
    elif is_document_file(file_path):
        pdf_path = None
        tmpdir = None
        reason = None
        try:
            pdf_path, tmpdir, reason = convert_document_to_pdf(file_path)
            if pdf_path:
                records = build_pdf_records_from_source(
                    embedder,
                    file_path,
                    pdf_path,
                    task,
                    low_memory,
                )
            else:
                raise RuntimeError(reason or "libreoffice_conversion_failed")
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()
    elif is_archive_file(file_path):
        records = build_archive_records(
            embedder,
            file_path,
            task,
            device,
            translate,
            low_memory,
            qwen_max_memory_mb,
            resolved_display_path,
            depth,
        )
        already_virtualized = True
    elif file_path.suffix.lower() == ".eml":
        records = build_eml_records(
            embedder,
            file_path,
            task,
            device,
            translate,
            low_memory,
            qwen_max_memory_mb,
            resolved_display_path,
            depth,
        )
        already_virtualized = True
    elif file_path.suffix.lower() == ".mbox":
        records = build_mbox_records(
            embedder,
            file_path,
            task,
            low_memory,
            resolved_display_path,
        )
    elif is_sqlite_file(file_path):
        records = build_sqlite_records(
            embedder,
            file_path,
            task,
            low_memory,
            resolved_display_path,
        )
    elif is_notebook_file(file_path):
        records = build_notebook_records(
            embedder,
            file_path,
            task,
            low_memory,
            resolved_display_path,
        )
    elif is_audio_file(file_path):
        records = build_audio_records(
            embedder,
            file_path,
            file_path,
            task,
            device,
            translate,
            low_memory,
            qwen_max_memory_mb,
        )
    elif is_cad_file(file_path):
        records = build_cad_records(
            embedder,
            file_path,
            task,
            low_memory,
        )
    elif is_image_file(file_path):
        records = build_image_records(
            embedder,
            file_path,
            task,
            low_memory,
        )
    else:
        text = read_text_file(file_path)
        if text is None:
            raise RuntimeError("unsupported_file_type")
        records = build_text_records(
            embedder,
            file_path,
            text,
            task,
            low_memory=low_memory,
        )

    if not already_virtualized and resolved_display_path != file_path.resolve().as_posix():
        records = virtualize_records(records, resolved_display_path)

    return records


def is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    return int(device.split(":", 1)[1])


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if is_cuda_device(requested):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-capable PyTorch device is available")
        if ":" in requested:
            try:
                requested_index = cuda_device_index(requested)
            except ValueError as exc:
                raise RuntimeError(f"Invalid CUDA device: {requested}") from exc
            if requested_index < 0 or requested_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {requested_index} was requested, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available"
                )

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not mps_backend.is_available():
            raise RuntimeError("MPS was requested, but no MPS-capable PyTorch device is available")

    if requested not in {"cpu", "mps"} and not is_cuda_device(requested):
        raise RuntimeError(f"Unsupported execution device: {requested}")

    return requested


def download_models() -> None:
    snapshot_download(JINA_MODEL_ID)
    snapshot_download(WHISPER_MODEL_ID)
    snapshot_download(QWEN_OMNI_MODEL_ID)
    if hub is None:
        raise RuntimeError("tensorflow_hub is required to download YAMNet")
    hub.load("https://tfhub.dev/google/yamnet/1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed local files or query text using Jina Embeddings V4.")
    parser.add_argument(
        "--task",
        default="retrieval",
        choices=["retrieval", "text-matching", "code"],
        help="Embedding task",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:N, or mps. Defaults to CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--query-text", help="Query string to embed for semantic search")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Pre-download Jina, Whisper, Qwen, and YAMNet into their normal cache directories",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="For non-English audio, run a second Whisper pass forced to English",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Keep only one large PyTorch model in VRAM at a time and force YAMNet onto CPU",
    )
    parser.add_argument(
        "--qwen-max-memory",
        type=int,
        help="Max VRAM for Qwen in MB (used with device_map=auto)",
    )
    parser.add_argument(
        "--document-extensions",
        help="Comma-separated LibreOffice document extensions passed by the Rust CLI for file ingest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.download_models and args.query_text is None and not args.document_extensions:
        raise RuntimeError("--document-extensions is required for file ingest")
    if args.download_models:
        download_models()
        return
    if args.document_extensions:
        extensions = {ext.strip().lower() for ext in args.document_extensions.split(",") if ext.strip()}
        if extensions:
            global DOCUMENT_EXTENSIONS
            DOCUMENT_EXTENSIONS = extensions
    if args.low_memory:
        disable_tf_gpu()
    device = detect_device(args.device)
    embedder = JinaEmbedder(device)

    if args.query_text is not None:
        ensure_jina_loaded(embedder, args.low_memory)
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
            for record in build_records_for_file(
                embedder,
                file_path,
                args.task,
                device,
                args.translate,
                args.low_memory,
                args.qwen_max_memory,
            ):
                emit_json(record)
            emit_json({"status": "done", "path": file_path.resolve().as_posix()})
        except Exception as exc:
            status = "error"
            reason = str(exc)
            if reason == "unsupported_file_type":
                status = "skipped"
            emit_json(
                {
                    "status": status,
                    "path": file_path.resolve().as_posix(),
                    "reason": reason,
                }
            )
            emit_json({"status": "done", "path": file_path.resolve().as_posix()})


if __name__ == "__main__":
    main()
