import torch
from transformers import AutoModel
from transformers.utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use flash_attention_2 if available (faster); otherwise sdpa (no extra deps)
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def load_nemotron_colembed_model():
    model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    ).eval()
    return model
