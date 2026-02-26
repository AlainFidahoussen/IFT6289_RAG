"""Jina v4 text-only retriever. Run with: uv sync && uv run python -m textual_retriever.model"""

import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_jina_v4_textual():
    """Load Jina-v4 for text-only retrieval via transformers."""
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to(device).eval()
    return model


if __name__ == "__main__":
    model = load_jina_v4_textual()
