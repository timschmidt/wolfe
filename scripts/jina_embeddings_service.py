#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import embed


class EmbeddingsRequest(BaseModel):
    model: str = Field(default=embed.JINA_MODEL_ID)
    input: str | list[str]
    task: str = Field(default="retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible Jina Embeddings V4 service.")
    parser.add_argument("--listen", default="127.0.0.1:18092", help="Listen address, formatted as HOST:PORT.")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, cuda, cuda:N, or mps.")
    parser.add_argument("--task", default="retrieval", help="Default Jina embedding task.")
    return parser.parse_args()


def create_app(device: str, default_task: str) -> FastAPI:
    resolved_device = embed.detect_device(device)
    jina = embed.JinaEmbedder(resolved_device)
    app = FastAPI(title="Wolfe Jina Embeddings", version="0.1.0")

    def encode_one(text: str, task: str) -> list[float]:
        embed.ensure_jina_loaded(jina, low_memory=False)
        model = jina.model
        if model is None:
            raise RuntimeError("embedding model failed to load")
        with embed.torch.inference_mode():
            embedding = model.encode_text(texts=text, task=task)
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().tolist()
        if embed.torch.cuda.is_available():
            embed.torch.cuda.empty_cache()
        return embedding

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": embed.JINA_MODEL_ID,
            "device": resolved_device,
            "loaded": jina.model is not None,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": embed.JINA_MODEL_ID,
                    "object": "model",
                    "owned_by": "jinaai",
                },
                {
                    "id": "wolfe-jina",
                    "object": "model",
                    "owned_by": "wolfe",
                },
            ],
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
        if request.model not in {embed.JINA_MODEL_ID, "wolfe-jina"}:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")

        values = request.input if isinstance(request.input, list) else [request.input]
        task = request.task or default_task
        started = time.monotonic()
        try:
            vectors = [encode_one(value, task) for value in values]
        except Exception as exc:
            if embed.torch.cuda.is_available():
                embed.torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt_tokens = sum(len(value.split()) for value in values)
        return {
            "object": "list",
            "model": request.model,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
            "geniehive": {
                "backend_model": embed.JINA_MODEL_ID,
                "device": resolved_device,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            },
        }

    return app


def main() -> None:
    args = parse_args()
    if ":" not in args.listen:
        raise SystemExit("--listen must be formatted as HOST:PORT")
    host, port_value = args.listen.rsplit(":", 1)
    app = create_app(args.device, args.task)
    uvicorn.run(app, host=host, port=int(port_value), log_level="info")


if __name__ == "__main__":
    main()
