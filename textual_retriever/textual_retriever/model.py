"""Jina v4 text-only retriever. Run with: uv sync && uv run python -m textual_retriever.model"""

import torch
from transformers import AutoModel
from transformers.utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

ZERANK2_MODEL_ID = "zeroentropy/zerank-2"


def load_jina_v4_textual():
    """Load Jina-v4 for text-only retrieval via transformers."""
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=ATTN_IMPL,
    )
    model = model.to(device).eval()
    return model


def load_zerank2():
    """Load zerank-2 cross-encoder reranker via sentence-transformers."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(
        ZERANK2_MODEL_ID,
        revision="refs/pr/2",
        trust_remote_code=True,
        model_kwargs={"dtype": torch.float16},
    )
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.eos_token_id
    return model


if __name__ == "__main__":
    model = load_jina_v4_textual()
