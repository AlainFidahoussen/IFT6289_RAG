"""DeepSeek-OCR-2 for document-to-markdown. Run with: uv sync && uv run python -m textual_extraction.model --image path/to/image.jpg --output path/to/output"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"
# DeepSeek-OCR-2 does not support SDPA; use eager when flash_attention_2 is unavailable
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else "eager"


def load_deepseek_ocr_2():

    model_name = 'deepseek-ai/DeepSeek-OCR-2'

    """Load DeepSeek-OCR-2 for text extraction via transformers."""
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_deepseek_ocr_2()
