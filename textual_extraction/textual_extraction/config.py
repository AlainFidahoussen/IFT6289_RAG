from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# VIDORE_SUBSET = "physics"  # "finance"
VIDORE_SUBSET = "computer_science"
VIDORE_LANG = "english"

CACHE_DIR_DEEPSEEK_MARKDOWNS = f"deepseek_cache_markdowns_{VIDORE_SUBSET}_{VIDORE_LANG}"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
