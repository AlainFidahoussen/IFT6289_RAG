import csv
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import typer

from textual_retriever.dataset import load_data_vidore
from textual_retriever.features import (
    load_deepseek_markdowns_from_disk,
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
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Rerank with zerank-2"),
    rerank_top_k: int = typer.Option(100, help="Number of dense candidates to rerank"),
    source: str = typer.Option("nemo", help="Markdown source: 'nemo' (dataset built-in) or 'deepseek'"),
    save_rankings: bool = typer.Option(False, "--save-rankings/--no-save-rankings", help="Save per-query top-k rankings to disk for answer generation"),
):
    """Evaluate NDCG@10 from precomputed embeddings only (no model load)."""
    cache_queries = f"jina_cache_queries_{subset}_{lang}"
    cache_markdowns = (
        f"jina_cache_markdowns_deepseek_{subset}_{lang}"
        if source == "deepseek"
        else f"jina_cache_markdowns_{subset}_{lang}"
    )

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

    reranker = None
    query_texts = None
    corpus_texts = None
    if rerank:
        from textual_retriever.model import load_zerank2
        logger.info("Loading zerank-2 reranker...")
        reranker = load_zerank2()
        query_id_to_text = dict(zip(ds_queries["query_id"], ds_queries["query"]))
        query_texts = [query_id_to_text[qid] for qid in query_ids_sorted]
        if source == "deepseek":
            logger.info("Loading DeepSeek-OCR-2 markdowns for reranking...")
            corpus_texts = load_deepseek_markdowns_from_disk(ds_corpus, subset, lang)
        else:
            corpus_texts = ds_corpus["markdown"]

    logger.info("Evaluating NDCG@10...")
    result = evaluate_ndcg(
        query_embeddings=query_embeddings_ordered,
        markdown_embeddings=markdown_embeddings,
        qrels=qrels,
        k=10,
        reranker=reranker,
        query_texts=query_texts,
        corpus_texts=corpus_texts,
        rerank_top_k=rerank_top_k,
        return_rankings=save_rankings,
    )
    if save_rankings:
        ndcg_at_10, rankings = result
    else:
        ndcg_at_10 = result

    model_name = "Jina-v4+zerank-2" if rerank else "Jina-v4"
    logger.success(
        f"Inference complete. Subset: {subset} - Language: {lang} - Source: {source} - NDCG@10: {ndcg_at_10:.1f}"
    )

    suffix = f"_{source}" if source != "nemo" else ""
    results_file = f"results_jina_reranked{suffix}.csv" if rerank else f"results_jina{suffix}.csv"
    write_header = not Path(results_file).exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model", "subset", "lang", "ndcg_at_10"])
        writer.writerow([datetime.now().isoformat(), model_name, subset, lang, f"{ndcg_at_10:.2f}"])
    logger.info(f"Result appended to {results_file}")

    if save_rankings:
        condition = f"jina_{source}_reranked" if rerank else f"jina_{source}"
        rankings_path = Path(f"data/processed/rankings_{condition}_{subset}_{lang}.json")
        rankings_path.parent.mkdir(parents=True, exist_ok=True)
        rankings_path.write_text(
            json.dumps({str(k): v for k, v in rankings.items()}, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Rankings saved to {rankings_path}")


if __name__ == "__main__":
    app()
