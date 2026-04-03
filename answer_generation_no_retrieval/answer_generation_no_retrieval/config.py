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

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATOR_MODEL = "qwen3.5:35b"
JUDGE_MODEL = "llama3.1:8b"

# Single condition: no retrieval (closed-book)
CONDITION_CLOSED_BOOK = "closed_book"
ALL_CONDITIONS = [CONDITION_CLOSED_BOOK]

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)
except ModuleNotFoundError:
    pass
