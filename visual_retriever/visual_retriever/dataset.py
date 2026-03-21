from datasets import load_dataset
from loguru import logger
import typer

from visual_retriever.features import (
    precompute_image_embeddings,
    precompute_query_embeddings,
)
from visual_retriever.model import load_visual_retriever_model
from visual_retriever.config import VIDORE_SUBSET, VIDORE_LANG

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
    cache_pages = f"colembed_cache_pages_{subset}_{lang}"
    cache_queries = f"colembed_cache_queries_{subset}_{lang}"

    logger.info("Loading model...")
    model = load_visual_retriever_model()

    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(subset=subset, lang=lang)

    logger.info("Pre-computing image embeddings...")
    precompute_image_embeddings(model, ds_corpus, save_dir=cache_pages)

    logger.info("Pre-computing query embeddings...")
    precompute_query_embeddings(model, ds_queries, save_dir=cache_queries)

    logger.success(f"Processing complete. Subset: {subset} - Language: {lang}")


if __name__ == "__main__":
    app()
