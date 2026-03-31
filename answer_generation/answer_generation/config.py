from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJ_ROOT.parent
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

VIDORE_SUBSET = "computer_science"
VIDORE_LANG = "english"

# Sibling subproject cache paths
TEXTUAL_CACHE_DIR = REPO_ROOT / "textual_retriever" / "data" / "processed"
VISUAL_CACHE_DIR = REPO_ROOT / "visual_retriever" / "data" / "processed"
DEEPSEEK_EXTRACTION_DIR = REPO_ROOT / "textual_extraction" / "data" / "processed"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATOR_MODEL = "qwen3.5:35b"
JUDGE_MODEL = "llama3.1:8b" # Smaller model, enough for textual judgment, different family than generator

TOP_K = 5
RERANK_TOP_K = 100

# Max characters per retrieved document sent to the generator.
# qwen3.5:35b has a 32k token context (~4 chars/token → ~128k chars total).
# 5 docs × 4000 chars = 20k chars — keeps prompts manageable for the 35b model.
# DeepSeek markdowns are verbose; 4000 chars captures the essential content.
MAX_CHARS_PER_DOC = 4_000

# Retrieval conditions
CONDITION_JINA_NEMO = "jina_nemo"
CONDITION_JINA_NEMO_RERANKED = "jina_nemo_reranked"
CONDITION_JINA_DEEPSEEK = "jina_deepseek"
CONDITION_JINA_DEEPSEEK_RERANKED = "jina_deepseek_reranked"
CONDITION_COLEMBED = "colembed"
CONDITION_HYBRID_NEMO = "hybrid_nemo"
CONDITION_HYBRID_DEEPSEEK = "hybrid_deepseek"

ALL_CONDITIONS = [
    CONDITION_JINA_NEMO,
    CONDITION_JINA_NEMO_RERANKED,
    CONDITION_JINA_DEEPSEEK,
    CONDITION_JINA_DEEPSEEK_RERANKED,
    CONDITION_COLEMBED,
    CONDITION_HYBRID_NEMO,
    CONDITION_HYBRID_DEEPSEEK,
]

# Modality derived from condition
CONDITION_TO_MODALITY: dict[str, str] = {
    CONDITION_JINA_NEMO: "text",
    CONDITION_JINA_NEMO_RERANKED: "text",
    CONDITION_JINA_DEEPSEEK: "text",
    CONDITION_JINA_DEEPSEEK_RERANKED: "text",
    CONDITION_COLEMBED: "image",
    CONDITION_HYBRID_NEMO: "hybrid",
    CONDITION_HYBRID_DEEPSEEK: "hybrid",
}

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)
except ModuleNotFoundError:
    pass
