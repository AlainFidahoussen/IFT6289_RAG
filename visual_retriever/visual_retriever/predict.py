import csv
from datetime import datetime
from loguru import logger
import typer

from visual_retriever.dataset import load_data_vidore
from visual_retriever.features import (
    load_precomputed_image_embeddings,
    load_precomputed_query_embeddings,
)
from visual_retriever.model import load_visual_retriever_model
from visual_retriever.utils import evaluate_ndcg
from visual_retriever.config import VIDORE_SUBSET, VIDORE_LANG

app = typer.Typer()


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

    logger.info("Loading pages embeddings...")
    pages_embeddings = load_precomputed_image_embeddings(ds_corpus, save_dir=cache_pages)

    logger.info("Loading query embeddings...")
    query_embeddings = load_precomputed_query_embeddings(ds_queries, save_dir=cache_queries)

    qrels = [
        {"query_id": q, "corpus_id": c, "score": s}
        for q, c, s in zip(ds_qrels["query_id"], ds_qrels["corpus_id"], ds_qrels["score"])
    ]

    # Build proper mapping and sort by query_id so evaluate_ndcg gets embeddings in the
    # same sorted order it expects (matches sorted(ground_truth_pages.keys()) in utils).
    query_id_to_emb = dict(zip(ds_queries["query_id"], query_embeddings))
    query_ids_sorted = sorted(set(q["query_id"] for q in qrels))
    query_embeddings_sorted = [query_id_to_emb[qid] for qid in query_ids_sorted]

    logger.info("Evaluating NDCG@10...")
    ndcg_at_k = evaluate_ndcg(
        model,  # need for get_scores()
        query_embeddings=query_embeddings_sorted,
        pages_embeddings=pages_embeddings,
        qrels=qrels,
        k=10,
    )

    logger.success(
        f"Inference complete. Subset: {subset} - Language: {lang} - NDCG@10: {ndcg_at_k:.1f}"
    )

    results_file = "results_colembed.csv"
    write_header = not __import__("pathlib").Path(results_file).exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model", "subset", "lang", "ndcg_at_10"])
        writer.writerow([datetime.now().isoformat(), "ColEmbed-3B-v2", subset, lang, f"{ndcg_at_k:.2f}"])
    logger.info(f"Result appended to {results_file}")


if __name__ == "__main__":
    app()
