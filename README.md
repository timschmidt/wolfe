# Wolfe
Multimodal semantic file search for intelligently investigating your files

Local only, 100% offline, your data stays on your computer.

# How it works
Wolfe uses LibreOffice, ffmpeg, mupdf, F3D, and a small set of structured-text/container extractors to decompose almost any file into streams of audio, text, and images. Audio streams are processed through YAMNet to identify audio events and type. When speech is detected, an additional processing step through Whisper Large v3 is used to transcribe and optionally translate it to text. When music is detected, an additional processing step through Qwen 2.5 Omni 7B is used to characterize and describe the music in text. CAD / 3D files supported by F3D are inspected for textual metadata and rendered into a fixed set of centered orthographic screenshots from the cardinal directions plus an isometric view. Archives are unpacked and recursively ingested through the same pipeline, email files extract message text and attachments, SQLite databases extract schema and sample rows, and notebooks extract markdown, code, and textual outputs. The resulting streams of images and text are passed through Jina Embedding v4 to generate the embeddings stored in the database with file metadata.

Search text is passed through Jina Embedding v4 to produce an embedding.  When queried with the search embedding, the database returns the closest matches and their associated metadata.

### Supported File Types

- Text: UTF-8 text files, including common plaintext formats such as `.cfg` `.ics` `.ini` `.jsonl` `.log` `.md` `.rst` `.srt` `.toml` `.vtt` `.yaml` `.yml` and other unknown-but-UTF-8 plaintext files
- MuPDF-backed paged documents: `.pdf` `.djvu` `.epub` `.ps`
- LibreOffice-converted documents: `.csv` `.dbf` `.dif` `.doc` `.docm` `.docx` `.dot` `.dotm` `.dotx` `.fodg` `.fodp` `.fods` `.fodt` `.htm` `.html` `.mht` `.mhtml` `.odb` `.odc` `.odf` `.odg` `.odm` `.odp` `.ods` `.odt` `.oth` `.otp` `.ots` `.ott` `.otg` `.otm` `.pot` `.potm` `.potx` `.pps` `.ppsm` `.ppsx` `.ppt` `.pptm` `.pptx` `.rtf` `.sda` `.sdc` `.sdd` `.sdw` `.slk` `.svg` `.sxc` `.sxd` `.sxg` `.sxi` `.sxm` `.sxw` `.tab` `.tsv` `.txt` `.uot` `.uop` `.uos` `.uof` `.vdx` `.vsd` `.vsdx` `.xhtml` `.xls` `.xlsm` `.xlsx` `.xlt` `.xltm` `.xltx` `.xml`
- CAD / 3D through F3D: `.3ds` `.gltf` `.glb` `.obj` `.ply` `.stl` `.vtm` `.vti` `.vtk` `.vtp` `.vtr` `.vts` `.vtu` `.wrl` `.x3d`
- Images: `.avif` `.bmp` `.gif` `.heic` `.heif` `.jpeg` `.jpg` `.png` `.psd` `.tif` `.tiff` `.webp`
- Audio: `.aac` `.aif` `.aiff` `.au` `.flac` `.m4a` `.mp3` `.ogg` `.opus` `.wav` `.webm`
- Video: `.3gp` `.avi` `.m2ts` `.m4v` `.mkv` `.mov` `.mp4` `.mpeg` `.mpg` `.ts` `.webm`
- Email / mailbox: `.eml` `.mbox`
- Notebook / database: `.ipynb` `.db3` `.sqlite` `.sqlite3`
- Archives and comic containers: `.zip` `.tar` `.tgz` `.tbz2` `.txz` `.7z` `.cbz` `.cbr`

### Ingestion Diagram

```mermaid
flowchart TD
    A[Input File] --> B{File Type}

    B -->|Text| T[Read Text File]
    T --> T1[Chunk Text]
    T1 --> T2[Embed Text Chunks]
    T2 --> T3[Store Text Records]

    B -->|PDF| P[Extract PDF Text]
    B -->|Document| D[Convert to PDF via LibreOffice]
    D --> P
    P --> P1[Chunk Text]
    P1 --> P2[Embed Text Chunks]
    P2 --> P3[Store Text Records]
    P --> P4[Render Pages to Images]
    P4 --> P5[Embed Page Images]
    P5 --> P6[Store Image Records]

    B -->|MuPDF Docs| M[Extract Text via MuPDF]
    M --> M1[Render Pages to Images]
    M --> M2[Embed Text Chunks]
    M2 --> M3[Store Text Records]
    M1 --> M4[Embed Page Images]
    M4 --> M5[Store Image Records]

    B -->|CAD| C[Load CAD through F3D]
    C --> C1[Extract F3D Text Metadata]
    C1 --> C2[Embed Metadata Text]
    C2 --> C3[Store CAD Text Records]
    C --> C4[Render Cardinal + Isometric Views]
    C4 --> C5[Embed View Images]
    C5 --> C6[Store CAD Image Records]

    B -->|Archive| AR[Extract Archive]
    AR --> AR1[Build Archive Summary Text]
    AR1 --> AR2[Embed Summary Text]
    AR2 --> AR3[Store Archive Summary Records]
    AR --> AR4[Recurse into Extracted Files]

    B -->|Email| E[Extract Headers and Body]
    E --> E1[Embed Email Text]
    E1 --> E2[Store Email Text Records]
    E --> E3[Extract Attachments]
    E3 --> E4[Recurse into Attachment Files]

    B -->|SQLite| S[Extract Schema and Sample Rows]
    S --> S1[Embed Database Text]
    S1 --> S2[Store Database Text Records]

    B -->|Notebook| N[Extract Markdown Code and Outputs]
    N --> N1[Embed Notebook Text]
    N1 --> N2[Store Notebook Text Records]

    B -->|Image| I[Load Image]
    I --> I1[Embed Image]
    I1 --> I2[Store Image Record]

    B -->|Audio| AU[Classify Audio Events through YAMNet]
    AU --> AU1[Build Audio Summary Text]
    AU1 --> AU2[Embed Summary Text]
    AU2 --> AU3[Store Audio Summary Records]
    AU --> AU4{Speech Detected?}
    AU4 -->|Yes| AU5[Whisper Transcription]
    AU5 --> AU6[Chunk Transcription]
    AU6 --> AU7[Embed Transcription Chunks]
    AU7 --> AU8[Store Audio Transcription Records]
    AU5 --> AU9{Non-English + --translate?}
    AU9 -->|Yes| AU10[Whisper Forced English]
    AU10 --> AU11[Chunk Translation]
    AU11 --> AU12[Embed Translation Chunks]
    AU12 --> AU13[Store Translation Records]
    AU --> AU14{Music Detected}
    AU14 -->|Yes| AU15[Qwen Omni Music Characterization]
    AU15 --> AU16[Chunk Description]
    AU16 --> AU17[Embed Description Chunks]
    AU17 --> AU18[Store Music Characterization Records]

    B -->|Video| V[Extract Subtitle Stream]
    V --> V1[Chunk Captions]
    V1 --> V2[Embed Caption Chunks]
    V2 --> V3[Store Caption Records]
    V --> V4[Extract Audio Track]
    V4 --> V5[Audio Pipeline through YAMNet + Whisper]
    V5 --> V6[Store Audio-Derived Records]
    V4 --> V10[Audio Music Characterization]
    V10 --> V11[Store Music Characterization Records]
    V --> V7[Extract Keyframes]
    V7 --> V8[Embed Keyframe Images]
    V8 --> V9[Store Keyframe Image Records]
```

## Usage

```bash
cargo run -- --path /path/to/input.txt
```

Optional:

```bash
cargo run -- --path /path/to/input-or-directory --task retrieval --python python3 --db wolfe.lance
cargo run -- --path /path/to/input-or-directory --translate --db wolfe.lance
cargo run -- --path /path/to/input-or-directory --db wolfe.lance
cargo run -- --path /path/to/input-or-directory --low-memory --db wolfe.lance
cargo run -- --path /path/to/input-or-directory --low-memory --qwen-max-memory 6000 --db wolfe.lance
```

Pre-download model caches:

```bash
cargo run -- --download-models
```

Search:

```bash
cargo run -- --search "error handling in rust" --db wolfe.lance --limit 10
cargo run -- --search "error handling in rust" --db wolfe.lance --range 10:20
cargo run -- --search "error handling in rust" --db wolfe.lance --json
```

By default, each CLI search starts the embedding helper long enough to vectorize
the query. For repeated search, run the Jina embedding model as a persistent
OpenAI-compatible service:

```bash
python scripts/jina_embeddings_service.py --listen 127.0.0.1:18092 --device cuda:1
cargo run -- --search "error handling in rust" --db wolfe.lance \
  --embedding-url http://127.0.0.1:18092/v1/embeddings \
  --embedding-model wolfe-jina
```

The service exposes `GET /health`, `GET /v1/models`, and
`POST /v1/embeddings`. It keeps the Jina weights loaded between requests, which
is much faster for browser or iterative CLI searches.

Web search UI:

```bash
cargo run -- --web --db wolfe.lance --listen 127.0.0.1:8767
```

Web mode can use the same persistent embedding service:

```bash
cargo run -- --web --db wolfe.lance --listen 127.0.0.1:8767 \
  --embedding-url http://127.0.0.1:18092/v1/embeddings \
  --embedding-model wolfe-jina
```

The web mode serves a local browser UI and JSON endpoints for semantic search
and neighboring context chunks:

- `GET /api/search?q=...&limit=20&offset=0`
- `GET /api/context?path=...&chunk=0&window=2`

Watch for changes:

```bash
cargo run -- --path /path/to/input-or-directory --watch --db wolfe.lance
```

To force a device explicitly:

```bash
cargo run -- --path /path/to/input-or-directory --device cuda
cargo run -- --path /path/to/input-or-directory --device cuda:1
```

CAD / 3D ingest example:

```bash
cargo run -- --path /path/to/cad-or-3d-directory --db wolfe.lance
```

CAD / 3D ingest with watch mode:

```bash
cargo run -- --path /path/to/cad-or-3d-directory --watch --db wolfe.lance
```

Archive ingest example:

```bash
cargo run -- --path /path/to/archive-or-directory --db wolfe.lance
```

Document ingestion for LibreOffice-supported formats requires `soffice` (LibreOffice) to be available on `PATH`. WordPerfect files (`.wp`, `.wp5`, `.wp6`, `.wpd`) are routed through this path when LibreOffice has writerperfect/libwpd support available.

Video ingestion requires `ffmpeg` and `ffprobe` to be available on `PATH`.

CAD ingestion requires `f3d` to be available on `PATH` with an offscreen rendering backend capable of writing `--output` images. Wolfe extracts text metadata from `f3d --no-render --verbose=debug` output and renders centered orthographic views from the front, back, left, right, top, bottom, and isometric directions. The isometric render uses a corner-facing camera aimed toward the model center. If F3D is installed but cannot render offscreen on your machine, CAD files will fail ingest with an F3D backend error.

Archive ingestion for `.zip`, `.tar`, `.tgz`, `.tbz2`, `.txz`, and `.cbz` uses Python’s built-in archive support. `.7z` requires `7z` on `PATH`, and `.cbr` uses `unrar` when available or falls back to `7z`.

Music characterization runs when YAMNet flags audio as music. Wolfe sends the audio to Qwen Omni and stores the response as additional audio-derived text chunks for search. The prompt used is:

```text
Fill out this profile about the music you hear. Be thorough.

Instrumentation/Vocals: ""
Soundscape: ""
Mood: ""
Genre: ""
Style: ""
Description: ""
Comment: ""
Progression: ""
Similar works: ""

(Progression means how the song evolves or if there are notable changes or moments.)
```

### CLI Options

- `-p, --path PATH`: File or directory to embed recursively (conflicts with `--search`).
- `--search TEXT`: Query string to vectorize and search semantically (conflicts with `--path`).
- `--web`: Run a local web UI for browsing semantic search results.
- `--listen ADDR`: Listen address for `--web` (default: `127.0.0.1:8767`).
- `--embedding-url URL`: OpenAI-compatible embedding endpoint for query embeddings.
- `--embedding-model NAME`: Model name sent to `--embedding-url` (default: `wolfe-jina`).
- `--embedding-api-key KEY`: API key sent as both Bearer auth and `X-Api-Key`; this can also be set with `WOLFE_EMBEDDING_API_KEY`.
- `--task TASK`: Embedding task name (default: `retrieval`).
- `--db PATH`: Path to the Lance table directory (default: `wolfe.lance`).
- `--python PATH`: Path to the Python interpreter (default: `python3`).
- `--device DEVICE`: Execution device (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`) (default: `auto`).
- `--script PATH`: Path to the embedding helper script (default: `scripts/embed.py`).
- `--limit N`: Maximum number of search results to return (default: `10`).
- `--range START:END`: Return a subset of search results (0-based, end-exclusive).
- `--json`: Emit search results as a JSON array instead of tab-separated text.
- `--translate`: For non-English audio, run a second Whisper pass forced to English.
- `--low-memory`: Keep only one large PyTorch model in VRAM at a time during ingest and force YAMNet onto CPU.
- `--qwen-max-memory MB`: Cap Qwen's GPU usage (MB) when `device_map=auto` is used; lower values offload more to CPU.
- `--ignore PATH`: File or directory name/path to ignore (repeatable).
- `--ignore-file FILE`: File containing newline-separated ignore entries.
- `--watch`: Watch for changes and keep the index up to date (requires `--path`).
- `--download-models`: Pre-download Jina, Whisper, Qwen, and YAMNet into their normal cache directories and exit.

### Persistent Embeddings and GenieHive

Wolfe's persistent embedding service is deliberately OpenAI-compatible so it
can be called directly or routed through GenieHive. In a GenieHive deployment,
register the Wolfe service as an embeddings backend with an upstream URL like
`http://127.0.0.1:18092/v1/embeddings` and an exposed model name such as
`wolfe-jina`. Wolfe can then point at the GenieHive control endpoint instead:

```bash
cargo run -- --web --db wolfe.lance \
  --embedding-url http://127.0.0.1:8800/v1/embeddings \
  --embedding-model wolfe-jina
```

If GenieHive requires an API key, set `WOLFE_EMBEDDING_API_KEY` or pass
`--embedding-api-key`. This keeps model residency, auth, observability, and
routing policy outside Wolfe while preserving the same search behavior.

Format notes for the CLI:

- There are no extra flags for archives, email, notebooks, or SQLite files; they are detected by extension and routed through the worker automatically.
- `--path` can point at a single archive, mailbox, notebook, database, or a directory containing any mix of supported formats.
- `--watch` re-runs ingest when supported source files change on disk. For archives, that means the archive file itself changing, not files previously extracted into a temp directory.
- `--ignore` and `--ignore-file` apply before recursive archive expansion, so ignored archives are skipped entirely.
- `--device` affects PyTorch model execution. `--low-memory` also changes the audio path by forcing YAMNet onto CPU; neither flag changes MuPDF extraction, SQLite reads, archive unpacking, or F3D rendering behavior.
- Archive members and email attachments are stored under virtual source paths in the index using a `container!/member/path` style path so search results still point back to the containing file.

## Setup

I would like for Wolfe to be implemented in pure Rust, but currently running the Jina Embeddings V4 model requires the use of a Python wrapper. Please file a PR or reach out if you know of a way to improve this. Until then:

### Create a Python venv and install deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pcre2
python -m pip install "transformers>=4.57,<5" pillow peft requests pymupdf numpy scipy soundfile tensorflow tensorflow-hub --no-build-isolation
```

Install F3D separately and ensure `f3d` is on `PATH` for CAD / 3D ingest.

Install `7z` if you want `.7z` or `.cbr` archive support. Install `unrar` if you want native `.cbr` extraction without the `7z` fallback.

Install a PyTorch build that matches your hardware:

- CPU fallback:

```bash
python -m pip install torch torchvision
```

- NVIDIA CUDA:

```bash
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

- Apple Silicon:

```bash
python -m pip install torch torchvision
```

The helper defaults to `--device auto`, which prefers CUDA, then MPS, then CPU.

Jina, Whisper, and Qwen are downloaded and cached automatically through Hugging Face the first time they are needed. YAMNet is downloaded and cached automatically through TensorFlow Hub. You can also pre-warm those caches with `cargo run -- --download-models`.

### Workaround for pcre

Set up this wrapper for pcre2 as pcre in the .venv

```Python
python - <<'PY'
import site
from pathlib import Path

site_dir = next(p for p in site.getsitepackages() if "site-packages" in p)
shim = Path(site_dir) / "pcre.py"
shim.write_text("from pcre2 import *\n")
print("Wrote shim:", shim)
PY
```

### Model caches

Wolfe uses the default library caches unless you override them with environment variables:

- Hugging Face: typically `~/.cache/huggingface/hub`
- TensorFlow Hub: typically `/tmp/tfhub_modules`

### Todo

- implement semantic boundary detection (sliding window?, llm based?)
- implement multi-threaded pdf decomposition and raster rust-side
- implement multi-threaded video decomposition rust-side
- implement multi-threaded document decomposition for LibreOffice, MS Office rust-side
