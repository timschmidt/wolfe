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
cargo run -- --path /path/to/input-or-directory --model-dir jina-embeddings-v4 --task retrieval --python python3
```

To force a device explicitly:

```bash
cargo run -- --path /path/to/input-or-directory --device cuda
```

When `--path` points at a directory, the CLI traverses it recursively and emits one JSON line per embedded file:

```json
{"path":"docs/readme.txt","modality":"text","embedding":[...]}
```

UTF-8 text files are embedded as text, and common image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`) are embedded through the model's image path. The Python helper stays alive for the whole run, so the model is loaded onto the selected device only once.
