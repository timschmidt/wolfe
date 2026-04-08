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
```

Search:

```bash
cargo run -- --search "error handling in rust" --db wolfe.lance --limit 10
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

Embeddings are stored in `wolfe.lance` by default. If `--db` ends with `.lance`, that path is treated as the final Lance table location; otherwise the table name defaults to `wolfe` under the given database directory. Each row includes the vector plus metadata such as absolute file path, file name, extension, parent directory, modality, chunk number, file size, and modified timestamp so search results can be mapped back to files. UTF-8 text files are embedded as text, common image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`) are embedded through the model's image path, PDFs are processed in both ways: embedded text is extracted and chunked first, then each rendered page is embedded as an image with chunk numbers continuing after the text chunks, common audio formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.webm`) are classified with YAMNet and embedded as text summaries of detected audio events. If YAMNet indicates speech or related vocalization categories, Wolfe also runs Whisper large-v3 and stores chunked embeddings of the transcription after the event-summary chunks.  Common video formats (`.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`, `.mpeg`, `.mpg`, `.ts`, `.webm`) are decomposed with `ffmpeg` into subtitle streams, extracted audio, and keyframes. Video subtitle text is chunked and embedded first, the extracted audio is then processed through the same YAMNet and optional Whisper transcription path, and keyframes are finally embedded through the image path with chunk numbers continuing after the text-derived chunks. Text files exceeding 20,000 tokens are chunked locally before embedding, and each chunk is stored as its own row. The Python helper stays alive for the whole run, so the model is loaded onto the selected device only once.

In search mode, the query string is embedded by the same Python model helper and searched against the stored vectors in LanceDB. Matching file paths and file names are printed to stdout as tab-separated lines.

In `--watch` mode on Linux, Wolfe uses the platform `notify` backend, which is `inotify`, to monitor the target path continuously. Changed and newly created files are reindexed, and removed files are deleted from the database. Existing records for a file are deleted before reindexing so stale chunk rows do not remain.

Video ingestion requires `ffmpeg` and `ffprobe` to be available on `PATH`.

### Todo

- implement database entry removal by metadata search
- implement inotify (and Win/Mac equivalents) support with --watch flag
- implement .ignore support for files you don't want indexed
- implement semantic boundary detection (sliding window?, llm based?)
- implement flag to show matching snippet for the search result items
