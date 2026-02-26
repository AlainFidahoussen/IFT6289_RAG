"""NDCG evaluation for dense (text) retrieval."""

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


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


def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().float().numpy()
    return np.asarray(x).astype(np.float32)


def evaluate_ndcg(
    query_embeddings: list[torch.Tensor],
    markdown_embeddings: list[torch.Tensor],
    qrels: Iterable[dict],
    k: int = 10,
) -> float:
    """Compute NDCG@k using precomputed query and markdown (dense) embeddings.

    Args:
        query_embeddings: List of query embedding tensors; same order as sorted query_ids from qrels.
        markdown_embeddings: List of corpus embedding tensors; index i = corpus position (corpus_id = i).
        qrels: Iterable of dicts with keys query_id, corpus_id, score (relevance 0/1/2).
        k: Rank cutoff for NDCG@k.

    Returns:
        Mean NDCG@k in 0–100 scale.
    """
    ground_truth_pages = _relevants_from_qrels(qrels)
    query_ids = sorted(ground_truth_pages.keys())
    query_id_to_embedding = dict(zip(query_ids, query_embeddings))

    # Stack corpus embeddings (N, dim), L2-normalize for cosine = dot
    corpus = np.stack([_to_numpy(e) for e in markdown_embeddings], axis=0).astype(np.float32)
    norm = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus = corpus / (norm + 1e-9)

    ndcg_scores = []
    for query_id in tqdm(ground_truth_pages, desc="NDCG@10"):
        q_emb = query_id_to_embedding[query_id]
        q = _to_numpy(q_emb)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        scores = (q @ corpus.T).flatten()
        top_k_positions = np.argsort(-scores)[:k]
        # Position i = corpus_id i (same order as markdown_embeddings)
        top_k_corpus_ids = [int(p) for p in top_k_positions]

        gt_pages = ground_truth_pages[query_id]
        relevance_at_rank = [gt_pages.get(cid, 0) for cid in top_k_corpus_ids]
        ndcg_scores.append(ndcg_at_k(relevance_at_rank, gt_pages, k=k))

    ndcg_at_10_mean = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    return ndcg_at_10_mean * 100
