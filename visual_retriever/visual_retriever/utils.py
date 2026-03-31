from collections import defaultdict
import math
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


def get_top_k(model, query_embedding, pages_embeddings, k=10):
    """Top-k indices between a query embedding and a list of page embeddings.


    Args:
        model: Model with get_scores(query_embedding, pages_embeddings) (e.g. Nemotron ColEmbed).
        query_embedding: [m, d] or [1, m, d].
        pages_embeddings: list of [T, d].
        k: Number of top indices to return.

    Returns:
        (k,) int array of indices where each element is the index of a page embedding in the list.
    """
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
    pages_embeddings: list[torch.Tensor],
    qrels: Iterable[dict],
    k: int = 10,
    return_rankings: bool = False,
):
    """Compute NDCG@k using precomputed query and page embeddings.

    Args:
        model: Model with get_scores(query_emb, pages_emb) (e.g. Nemotron ColEmbed).
        query_embeddings: List of query embedding tensors; same order as query_ids.
        qrels: Iterable of dicts with keys query_id, corpus_id, score (relevance 0/1/2).
        pages_embeddings: List of page embedding tensors; index i = corpus position i.
        k: Rank cutoff for NDCG@k.
        return_rankings: If True, also return per-query top-k rankings as dict[query_id, list[corpus_id]].

    Returns:
        Mean NDCG@k in 0–100 scale, or (ndcg, rankings) if return_rankings is True.
    """

    ground_truth_pages = _relevants_from_qrels(qrels)

    query_ids = sorted(ground_truth_pages.keys())
    query_id_to_embedding = dict(zip(query_ids, query_embeddings))

    ndcg_scores = []
    rankings: dict[int, list[int]] = {}
    for query_id in tqdm(ground_truth_pages, desc="NDCG@10"):
        query_embedding = query_id_to_embedding[query_id]
        gt_pages = ground_truth_pages[query_id]
        top_k_indices = get_top_k(model, query_embedding, pages_embeddings, k=k)
        torch.cuda.empty_cache()
        rankings[query_id] = [int(i) for i in top_k_indices]
        relevance_at_rank = [gt_pages.get(idx, 0) for idx in top_k_indices]
        ndcg_scores.append(ndcg_at_k(relevance_at_rank, gt_pages, k=k))

    ndcg_at_10_mean = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    ndcg = ndcg_at_10_mean * 100
    return (ndcg, rankings) if return_rankings else ndcg
