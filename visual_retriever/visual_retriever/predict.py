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
from visual_retriever.config import CACHE_DIR_IMAGE_EMBEDDINGS, CACHE_DIR_QUERY_EMBEDDINGS

app = typer.Typer()


@app.command()
def main():
    logger.info("Loading model...")
    model = load_visual_retriever_model()

    logger.info("Loading dataset...")
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(
        subset=VIDORE_SUBSET, lang=VIDORE_LANG
    )

    logger.info("Loading pages embeddings...")
    pages_embeddings = load_precomputed_image_embeddings(ds_corpus, save_dir=CACHE_DIR_IMAGE_EMBEDDINGS)

    logger.info("Loading query embeddings...")
    query_embeddings = load_precomputed_query_embeddings(ds_queries, save_dir=CACHE_DIR_QUERY_EMBEDDINGS)

    qrels = [
        {"query_id": q, "corpus_id": c, "score": s}
        for q, c, s in zip(ds_qrels["query_id"], ds_qrels["corpus_id"], ds_qrels["score"])
    ]

    logger.info("Evaluating NDCG@10...")
    ndcg_at_k = evaluate_ndcg(
        model,  # need for get_scores()
        query_embeddings=query_embeddings,
        pages_embeddings=pages_embeddings,
        qrels=qrels,
        k=10,
    )

    logger.success(
        f"Inference complete. Subset: {VIDORE_SUBSET} - Language: {VIDORE_LANG} - NDCG@10: {ndcg_at_k:.1f}"
    )


if __name__ == "__main__":
    app()
