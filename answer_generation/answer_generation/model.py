"""Ollama client for answer generation and LLM-as-judge."""

import base64
import io
import time

import httpx
from loguru import logger
from PIL.Image import Image

from answer_generation.config import OLLAMA_BASE_URL


class OllamaClient:
    """Thin wrapper around the Ollama REST /api/chat endpoint."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url

    def chat(
        self,
        model: str,
        prompt: str,
        images: list[Image] | None = None,
        response_format: str | None = None,
    ) -> str:
        """Send a single-turn chat request and return the response text.

        A fresh HTTP connection is created per call to avoid stale persistent
        connections causing timeouts on long-running batch jobs.

        Args:
            model: Ollama model name (e.g. "qwen3.5:latest").
            prompt: User message text.
            images: Optional PIL images to attach (for multimodal models).
            response_format: Pass "json" to force JSON output mode.
        """
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [self._encode_image(img) for img in images]

        payload: dict = {
            "model": model,
            "messages": [message],
            "stream": False,
            "keep_alive": -1,
            "think": False,
            "options": {"temperature": 0, "num_predict": 1024},
        }
        if response_format == "json":
            payload["format"] = "json"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=1200.0) as http:
                    response = http.post(
                        f"{self._base_url}/api/chat",
                        json=payload,
                    )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama request failed, retrying",
                    attempt=attempt + 1,
                    model=model,
                    error=str(exc),
                )
                time.sleep(2**attempt)

        raise RuntimeError(
            f"Ollama request failed after 3 attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _encode_image(image: Image) -> str:
        """Base64-encode a PIL image as PNG for Ollama's images field."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def load_zerank2():
    """Load the zerank-2 cross-encoder reranker (sentence-transformers)."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading zerank-2 reranker...")
    return CrossEncoder("zeroentropy/zerank-2", max_length=8192)
