import torch
from transformers import AutoModel
from transformers.utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def load_visual_retriever_model():
    """Load ColEmbed visual retriever (image + text)."""
    model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    )
    model = model.to(device).eval()
    return model


if __name__ == "__main__":
    model = load_visual_retriever_model()
