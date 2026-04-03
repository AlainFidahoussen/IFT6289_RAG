"""Ollama client for closed-book answer generation and LLM-as-judge."""

import time

import httpx
from loguru import logger

from answer_generation_no_retrieval.config import OLLAMA_BASE_URL


class OllamaClient:
    """Thin wrapper around the Ollama REST /api/chat endpoint."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url

    def chat(
        self,
        model: str,
        prompt: str,
        response_format: str | None = None,
    ) -> str:
        """Send a single-turn chat request and return the response text.

        Args:
            model: Ollama model name (e.g. "qwen3.5:latest").
            prompt: User message text.
            response_format: Pass "json" to force JSON output mode.
        """
        message: dict = {"role": "user", "content": prompt}

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
