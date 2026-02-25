from collections import defaultdict
import math
from typing import Iterable

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

from ift6289_rag.features import load_precomputed_image_embeddings


def get_top_k(model, query_embedding, pages_embeddings, k=10):
    """Top-k indices using model.get_scores (handles padding, device, same as official eval)."""
    # query_embedding: [m, d] or [1, m, d]; pages_embeddings: list of [T, d]
    scores = model.get_scores(query_embedding, pages_embeddings)
    scores = scores[0].cpu().float().numpy()  # [num_queries, num_passages] -> [num_passages]
    return np.argsort(scores)[::-1][:k]


def ndcg_at_k(relevance_at_rank: list, rel_dict: dict, k: int = 10) -> float:
    """Compute NDCG@k with graded relevance (ViDoRe: 0, 1, 2). IDCG from full rel_dict."""
    rel = relevance_at_rank[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_relevances = sorted(rel_dict.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_relevances))
    return dcg / idcg if idcg > 0 else 0.0


def _relevants_from_qrels(qrels: Iterable[dict]) -> dict[int, dict[int, int]]:
    """Build query_id -> {corpus_id: score} from qrels rows."""
    relevants = defaultdict(dict)
    for r in qrels:
        relevants[r["query_id"]][r["corpus_id"]] = r["score"]
    return dict(relevants)


def evaluate_ndcg(
    model,
    query_embeddings: list[torch.Tensor],
    query_ids: list[int],
    qrels: Iterable[dict],
    pages_embeddings: list[torch.Tensor],
    idx_to_corpus_id: list[int],
    k: int = 10,
):
    """Compute NDCG@k using precomputed query and page embeddings.

    Args:
        model: Model with get_scores(query_emb, pages_emb) (e.g. Nemotron ColEmbed).
        query_embeddings: List of query embedding tensors; same order as query_ids.
        query_ids: List of query_id for each embedding (same length as query_embeddings).
        qrels: Iterable of dicts with keys query_id, corpus_id, score (relevance 0/1/2).
        pages_embeddings: List of page embedding tensors; index i = corpus position i.
        idx_to_corpus_id: idx_to_corpus_id[i] = corpus_id of the i-th page (same order as pages_embeddings).
        k: Rank cutoff for NDCG@k.

    Returns:
        Mean NDCG@k in 0–100 scale.
    """
    query_id_to_embedding = dict(zip(query_ids, query_embeddings))
    relevants = _relevants_from_qrels(qrels)

    ndcg_scores = []
    for query_id in tqdm(relevants, desc="NDCG@10"):
        query_embedding = query_id_to_embedding.get(query_id)
        rel_dict = relevants[query_id]
        top_k_indices = get_top_k(model, query_embedding, pages_embeddings, k=k)
        relevance_at_rank = [rel_dict.get(idx_to_corpus_id[idx], 0) for idx in top_k_indices]
        ndcg_scores.append(ndcg_at_k(relevance_at_rank, rel_dict, k=k))

    ndcg_at_10_mean = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    return ndcg_at_10_mean * 100
