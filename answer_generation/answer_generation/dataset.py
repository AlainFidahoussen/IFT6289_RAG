"""ViDoRe v3 data loading for answer generation."""

from datasets import Dataset, load_dataset
from loguru import logger

from answer_generation.config import VIDORE_SUBSET, VIDORE_LANG


def load_data_vidore(
    subset: str = VIDORE_SUBSET,
    lang: str = VIDORE_LANG,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load ViDoRe v3 corpus, queries (language-filtered), and qrels."""
    dataset_name = f"vidore/vidore_v3_{subset}"

    ds_corpus = load_dataset(dataset_name, "corpus", split="test")
    ds_queries_full = load_dataset(dataset_name, "queries", split="test")
    ds_qrels_full = load_dataset(dataset_name, "qrels", split="test")

    ds_queries = ds_queries_full.filter(lambda x: x["language"] == lang)
    query_ids = set(ds_queries["query_id"])
    ds_qrels = ds_qrels_full.filter(lambda x: x["query_id"] in query_ids)

    logger.info(
        "Dataset loaded",
        subset=subset,
        lang=lang,
        num_queries=len(ds_queries),
        num_corpus=len(ds_corpus),
    )
    return ds_corpus, ds_queries, ds_qrels


def get_answer_field(ds_queries: Dataset) -> str:
    """Detect the ground-truth answer field name in the queries split."""
    for candidate in ("answer", "answers", "gold_answer", "reference_answer"):
        if candidate in ds_queries.column_names:
            return candidate
    raise ValueError(
        f"No answer field found. Available columns: {ds_queries.column_names}"
    )
