from datasets import load_dataset
from loguru import logger
import typer

from textual_extraction.config import VIDORE_SUBSET, VIDORE_LANG
from textual_extraction.model import load_deepseek_ocr_2
from textual_extraction.features import precompute_deepseek_markdowns

app = typer.Typer()


def load_data_vidore(subset: str = VIDORE_SUBSET, lang: str = VIDORE_LANG):
    """Load ViDoRe v3 corpus, queries, qrels, and documents_metadata."""
    dataset_name = f"vidore/vidore_v3_{subset}"

    ds_corpus = load_dataset(dataset_name, "corpus", split="test")
    ds_queries_full = load_dataset(dataset_name, "queries", split="test")
    ds_qrels_full = load_dataset(dataset_name, "qrels", split="test")
    ds_metadata = load_dataset(dataset_name, "documents_metadata", split="test")

    ds_queries = ds_queries_full.filter(lambda x: x["language"] == lang)
    query_ids = set(ds_queries["query_id"])
    ds_qrels = ds_qrels_full.filter(lambda x: x["query_id"] in query_ids)

    return ds_corpus, ds_queries, ds_qrels, ds_metadata


@app.command()
def main(
    subset: str = typer.Option(VIDORE_SUBSET, help="ViDoRe v3 subset name"),
    lang: str = typer.Option(VIDORE_LANG, help="Query language filter"),
):
    cache_dir = f"deepseek_cache_markdowns_{subset}_{lang}"

    logger.info("Loading model...")
    model, tokenizer = load_deepseek_ocr_2()

    logger.info(f"Loading dataset: {subset} ({lang})...")
    ds_corpus, _, _, _ = load_data_vidore(subset=subset, lang=lang)

    logger.info("Extracting markdown with DeepSeek-OCR-2...")
    precompute_deepseek_markdowns(model, tokenizer, ds_corpus, save_dir=cache_dir)

    logger.success(f"Extraction complete. Subset: {subset} - Language: {lang}")


if __name__ == "__main__":
    app()
