from loguru import logger
import typer

from ift6289_rag.dataset import load_data_vidore
from ift6289_rag.features import load_precomputed_image_embeddings, load_precomputed_query_embeddings
from ift6289_rag.model import load_nemotron_colembed_model
from ift6289_rag.utils import evaluate_ndcg

app = typer.Typer()


@app.command()
def main():
    logger.info("Loading model...")
    model = load_nemotron_colembed_model()

    logger.info("Loading dataset...")
    subset = "physics"
    lang = "english"
    ds_corpus, ds_queries, ds_qrels, ds_metadata = load_data_vidore(subset, lang)

    logger.info("Loading pages embeddings...")
    pages_embeddings = load_precomputed_image_embeddings(ds_corpus)

    logger.info("Loading query embeddings...")
    query_embeddings = load_precomputed_query_embeddings(ds_queries)
    query_ids = list(ds_queries["query_id"])

    idx_to_corpus_id = [ds_corpus[i]["corpus_id"] for i in range(len(ds_corpus))]
    qrels = [{"query_id": q, "corpus_id": c, "score": s} for q, c, s in zip(ds_qrels["query_id"], ds_qrels["corpus_id"], ds_qrels["score"])]

    logger.info("Evaluating NDCG@10...")
    ndcg_at_k = evaluate_ndcg(
        model,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        qrels=qrels,
        pages_embeddings=pages_embeddings,
        idx_to_corpus_id=idx_to_corpus_id,
        k=10,
    )

    logger.success(
        f"Inference complete. Subset: {subset} - Language: {lang} - NDCG@10: {ndcg_at_k:.1f}"
    )


if __name__ == "__main__":
    app()
