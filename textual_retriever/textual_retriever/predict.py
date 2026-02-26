from loguru import logger
import typer

from textual_retriever.dataset import load_data_vidore
from textual_retriever.features import (
    load_precomputed_markdown_embeddings,
    load_precomputed_query_embeddings,
)
from textual_retriever.utils import evaluate_ndcg
from textual_retriever.config import VIDORE_SUBSET, VIDORE_LANG
from textual_retriever.config import CACHE_DIR_QUERY_EMBEDDINGS, CACHE_DIR_MARKDOWN_EMBEDDINGS

app = typer.Typer()


@app.command()
def main():
    """Evaluate NDCG@10 from precomputed embeddings only (no model load)."""
    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(VIDORE_SUBSET, VIDORE_LANG)

    logger.info("Loading markdown embeddings...")
    markdown_embeddings = load_precomputed_markdown_embeddings(ds_corpus, save_dir=CACHE_DIR_MARKDOWN_EMBEDDINGS)

    logger.info("Loading query embeddings...")
    query_embeddings = load_precomputed_query_embeddings(ds_queries, save_dir=CACHE_DIR_QUERY_EMBEDDINGS)

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
        f"Inference complete. Subset: {VIDORE_SUBSET} - Language: {VIDORE_LANG} - NDCG@10: {ndcg_at_10:.1f}"
    )


if __name__ == "__main__":
    app()
