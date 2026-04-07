# embedding

Minimal Rust CLI that embeds a local file using Jina Embeddings V4 by invoking a local Python helper.

## Setup

### Create a Python venv and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "transformers>=4.52,<5" torchvision torch pillow peft requests
```

### Ensure model files are present

```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/jinaai/jina-embeddings-v4
```

or pass `--model-dir`.

## Usage

```bash
cargo run -- --file /path/to/input.txt
```

Optional:

```bash
cargo run -- --file /path/to/input.txt --model-dir jina-embeddings-v4 --task retrieval --python python3
```
