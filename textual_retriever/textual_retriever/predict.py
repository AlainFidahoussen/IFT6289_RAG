import csv
from datetime import datetime
from loguru import logger
import typer

from textual_retriever.dataset import load_data_vidore
from textual_retriever.features import (
    load_precomputed_markdown_embeddings,
    load_precomputed_query_embeddings,
)
from textual_retriever.utils import evaluate_ndcg
from textual_retriever.config import VIDORE_SUBSET, VIDORE_LANG

app = typer.Typer()


@app.command()
def main(
    subset: str = typer.Option(VIDORE_SUBSET, help="ViDoRe v3 subset name"),
    lang: str = typer.Option(VIDORE_LANG, help="Query language filter"),
):
    """Evaluate NDCG@10 from precomputed embeddings only (no model load)."""
    cache_queries = f"jina_cache_queries_{subset}_{lang}"
    cache_markdowns = f"jina_cache_markdowns_{subset}_{lang}"

    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(subset, lang)

    logger.info("Loading markdown embeddings...")
    markdown_embeddings = load_precomputed_markdown_embeddings(ds_corpus, save_dir=cache_markdowns)

    logger.info("Loading query embeddings...")
    query_embeddings = load_precomputed_query_embeddings(ds_queries, save_dir=cache_queries)

    query_id_to_embedding = dict(zip(ds_queries["query_id"], query_embeddings))

    qrels = [
        {"query_id": q, "corpus_id": c, "score": s}
        for q, c, s in zip(ds_qrels["query_id"], ds_qrels["corpus_id"], ds_qrels["score"])
    ]

    # evaluate_ndcg expects query_embeddings in sorted(query_id) order
    query_ids_sorted = sorted(set(q["query_id"] for q in qrels))
    query_embeddings_ordered = [query_id_to_embedding[qid] for qid in query_ids_sorted]

    logger.info("Evaluating NDCG@10...")
    ndcg_at_10 = evaluate_ndcg(
        query_embeddings=query_embeddings_ordered,
        markdown_embeddings=markdown_embeddings,
        qrels=qrels,
        k=10,
    )

    logger.success(
        f"Inference complete. Subset: {subset} - Language: {lang} - NDCG@10: {ndcg_at_10:.1f}"
    )

    results_file = "results_jina.csv"
    write_header = not __import__("pathlib").Path(results_file).exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model", "subset", "lang", "ndcg_at_10"])
        writer.writerow([datetime.now().isoformat(), "Jina-v4", subset, lang, f"{ndcg_at_10:.2f}"])
    logger.info(f"Result appended to {results_file}")


if __name__ == "__main__":
    app()
