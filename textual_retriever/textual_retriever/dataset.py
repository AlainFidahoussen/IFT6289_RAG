"""ViDoRe v3 data loading (shared with textual retriever)."""

from datasets import load_dataset
from loguru import logger
import typer

from textual_retriever.model import load_jina_v4_textual
from textual_retriever.features import (
    load_deepseek_markdowns_from_disk,
    precompute_markdown_embeddings,
    precompute_query_embeddings,
)
from textual_retriever.config import VIDORE_SUBSET, VIDORE_LANG

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
    source: str = typer.Option("nemo", help="Markdown source: 'nemo' (dataset built-in) or 'deepseek'"),
):
    cache_queries = f"jina_cache_queries_{subset}_{lang}"
    cache_markdowns = (
        f"jina_cache_markdowns_deepseek_{subset}_{lang}"
        if source == "deepseek"
        else f"jina_cache_markdowns_{subset}_{lang}"
    )

    logger.info("Loading model...")
    model = load_jina_v4_textual()

    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(subset=subset, lang=lang)

    logger.info("Pre-computing query embeddings...")
    precompute_query_embeddings(model, ds_queries, save_dir=cache_queries)

    markdown_texts = None
    if source == "deepseek":
        logger.info("Loading DeepSeek-OCR-2 markdowns from disk...")
        markdown_texts = load_deepseek_markdowns_from_disk(ds_corpus, subset, lang)

    logger.info(f"Pre-computing markdown embeddings (source={source})...")
    precompute_markdown_embeddings(model, ds_corpus, save_dir=cache_markdowns, markdown_texts=markdown_texts)

    logger.success(f"Processing complete. Subset: {subset} - Language: {lang} - Source: {source}")


if __name__ == "__main__":
    app()
