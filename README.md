# Wolfe

Multimodal semantic file search using Jina Embeddings V4, YAMNet, and Whisper Large V3

For intelligently investigating your files

Local only, 100% offline, your data stays on your computer.

## Setup

I would like for Wolfe to be implemented in pure Rust, but currently running the Jina Embeddings V4 model requires the use of a Python wrapper.  Please file a PR or reach out if you know of a way to improve this.  Until then:

### Create a Python venv and install deps

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "transformers>=4.52,<5" pillow peft requests pymupdf numpy scipy soundfile tensorflow tensorflow-hub
```

Install a PyTorch build that matches your hardware:

- CPU fallback:

```bash
python -m pip install torch torchvision
```

- NVIDIA CUDA:

```bash
python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

- Apple Silicon:

```bash
python -m pip install torch torchvision
```

The helper defaults to `--device auto`, which prefers CUDA, then MPS, then CPU.

### Ensure model files are present

```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/jinaai/jina-embeddings-v4
```

or pass `--model-dir`

## Usage

```bash
cargo run -- --path /path/to/input.txt
```

Optional:

```bash
cargo run -- --path /path/to/input-or-directory --model-dir jina-embeddings-v4 --task retrieval --python python3 --db wolfe.lance
cargo run -- --path /path/to/input-or-directory --translate --db wolfe.lance
```

Search:

```bash
cargo run -- --search "error handling in rust" --db wolfe.lance --limit 10
cargo run -- --search "error handling in rust" --db wolfe.lance --range 10:20
cargo run -- --search "error handling in rust" --db wolfe.lance --json
```

Watch for changes:

```bash
cargo run -- --path /path/to/input-or-directory --watch --db wolfe.lance
```

To force a device explicitly:

```bash
cargo run -- --path /path/to/input-or-directory --device cuda
```

When `--path` points at a directory, the CLI traverses it recursively

You can exclude content from both recursive ingest and `--watch` with repeated `--ignore` arguments or with `--ignore-file path/to/list.txt`. Ignore entries may be file or directory names such as `node_modules` or `target`, or explicit relative/absolute paths. Any file with a matching name and anything under any directory with a matching name or path is skipped.

Embeddings are stored in `wolfe.lance` by default. If `--db` ends with `.lance`, that path is treated as the final Lance table location; otherwise the table name defaults to `wolfe` under the given database directory. Each row includes the vector plus metadata such as absolute file path, file name, extension, parent directory, modality, chunk number, offset, plaintext, file size, and modified timestamp so search results can be mapped back to files. `offset` is stored in bytes for plain text files, seconds for audio-derived records, pages for PDF-derived records, and frames for video-derived records. `plaintext` stores the text input used to create that chunk's embedding when the modality is text-derived, including plain text, PDF text, audio event summaries, transcriptions, and video caption/audio-derived chunks. UTF-8 text files are embedded as text, common image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`) are embedded through the model's image path, PDFs are processed in both ways: embedded text is extracted and chunked first, then each rendered page is embedded as an image with chunk numbers continuing after the text chunks, common audio formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.webm`) are classified with YAMNet and embedded as text summaries of detected audio events. If YAMNet indicates speech or related vocalization categories, Wolfe also runs Whisper large-v3 and stores chunked embeddings of the transcription after the event-summary chunks. When `--translate` is enabled, Wolfe performs a second Whisper pass forced to English for non-English audio and stores those translated chunks immediately after the native-language transcription chunks. Common video formats (`.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`, `.mpeg`, `.mpg`, `.ts`, `.webm`) are decomposed with `ffmpeg` into subtitle streams, extracted audio, and keyframes. Video subtitle text is chunked and embedded first, the extracted audio is then processed through the same YAMNet and optional Whisper transcription path, and keyframes are finally embedded through the image path with chunk numbers continuing after the text-derived chunks. Text files exceeding 20,000 tokens are chunked locally before embedding, and each chunk is stored as its own row. The Python helper stays alive for the whole run, so the model is loaded onto the selected device only once.

In search mode, the query string is embedded by the same Python model helper and searched against the stored vectors in LanceDB. Search results are printed to stdout as tab-separated columns containing the matching path, file name, modality, stored locator (`byte`, `second`, `page`, or `frame`), and the stored `plaintext` snippet for that chunk when available. Use `--range START:END` (0-based, end-exclusive) to return only a subset of results, and `--json` to emit a JSON array instead of tab-separated text. This avoids rerunning Whisper, PDF extraction, or other expensive processing during search.

In `--watch` mode on Linux, Wolfe uses the platform `notify` backend, which is `inotify`, to monitor the target path continuously. Changed and newly created files are reindexed, and removed files are deleted from the database. Existing records for a file are deleted before reindexing so stale chunk rows do not remain. The same ignore rules from `--ignore` and `--ignore-file` are applied to watch events before reindexing.

Video ingestion requires `ffmpeg` and `ffprobe` to be available on `PATH`.

### Ingestion Diagram

See `docs/ingestion-diagram.md` for a diagram of the ingestion pathways by file type.

### Todo

- implement semantic boundary detection (sliding window?, llm based?)
- implement ingestion progress indicator
