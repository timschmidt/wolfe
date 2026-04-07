# embedding

Minimal Rust CLI that embeds a file or recursively embeds a directory using Jina Embeddings V4 through a persistent local Python worker.

## Setup

### Create a Python venv and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "transformers>=4.52,<5" pillow peft requests
```

Install a PyTorch build that matches your hardware:

- CPU fallback:

```bash
python -m pip install torch torchvision
```

- NVIDIA CUDA:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

- Apple Silicon:

```bash
python -m pip install torch torchvision
```

The helper now defaults to `--device auto`, which prefers CUDA, then MPS, then CPU.

### Ensure model files are present

```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/jinaai/jina-embeddings-v4
```

or pass `--model-dir`.

## Usage

```bash
cargo run -- --path /path/to/input.txt
```

Optional:

```bash
cargo run -- --path /path/to/input-or-directory --model-dir jina-embeddings-v4 --task retrieval --python python3 --db wolfe.lance
```

To force a device explicitly:

```bash
cargo run -- --path /path/to/input-or-directory --device cuda
```

When `--path` points at a directory, the CLI traverses it recursively and writes embeddings into LanceDB from the Rust process:

```json
{"db":"/abs/path/to/wolfe.lance","table":"wolfe","stored":128,"skipped":4,"errors":0}
```

Embeddings are stored in `wolfe.lance` by default. If `--db` ends with `.lance`, that path is treated as the final Lance table location; otherwise the table name defaults to `embeddings` under the given database directory. Each row includes the vector plus metadata such as absolute file path, file name, extension, parent directory, modality, file size, and modified timestamp so search results can be mapped back to files. UTF-8 text files are embedded as text, and common image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`) are embedded through the model's image path. The Python helper stays alive for the whole run, so the model is loaded onto the selected device only once, but LanceDB persistence now happens on the Rust side.
