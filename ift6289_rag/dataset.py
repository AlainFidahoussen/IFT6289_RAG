from datasets import load_dataset
from loguru import logger
import typer

from ift6289_rag.features import precompute_image_embeddings, precompute_query_embeddings
from ift6289_rag.model import load_nemotron_colembed_model

app = typer.Typer()


def load_data_vidore(subset: str = "physics", lang: str = "english"):
    """Load ViDoRe v3 corpus, queries, qrels, and documents_metadata for a given subset and language.

    Args:
        subset: Dataset subset (e.g. "physics", "computer_science", "energy").
        lang: Query language to keep ("english", "french", "german", "italian", "portuguese", "spanish").

    Returns:
        Tuple of (ds_corpus, ds_queries, ds_qrels, ds_metadata), all with split='test'.
        Queries and qrels are filtered to the requested language.
    """
    dataset_name = f"vidore/vidore_v3_{subset}"

    ds_corpus = load_dataset(dataset_name, "corpus", split="test")
    ds_queries_full = load_dataset(dataset_name, "queries", split="test")
    ds_qrels_full = load_dataset(dataset_name, "qrels", split="test")
    ds_metadata = load_dataset(dataset_name, "documents_metadata", split="test")

    # Filter queries by language (dataset uses "language" field, e.g. "english", "french")
    ds_queries = ds_queries_full.filter(lambda x: x["language"] == lang)
    query_ids = set(ds_queries["query_id"])

    # Filter qrels to only include (query_id, corpus_id, score) for the selected queries
    ds_qrels = ds_qrels_full.filter(lambda x: x["query_id"] in query_ids)

    return ds_corpus, ds_queries, ds_qrels, ds_metadata


@app.command()
def main():

    logger.info("Loading model...")
    model = load_nemotron_colembed_model()

    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore()

    logger.info("Pre-computing image embeddings...")
    save_dir = "colembed_cache_pages_fp16"
    precompute_image_embeddings(model, ds_corpus, save_dir=save_dir)

    logger.info("Pre-computing query embeddings...")
    save_dir = "colembed_cache_queries_fp16"
    precompute_query_embeddings(model, ds_queries, save_dir=save_dir)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
