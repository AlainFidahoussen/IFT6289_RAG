"""NDCG evaluation for dense (text) retrieval."""

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


def rerank_candidates(
    reranker,
    query_text: str,
    candidate_texts: list[str],
    batch_size: int = 1,
) -> np.ndarray:
    """Score (query, passage) pairs with a CrossEncoder reranker."""
    pairs = [[query_text, t] for t in candidate_texts]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)


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
    reranker=None,
    query_texts: list[str] | None = None,
    corpus_texts: list[str] | None = None,
    rerank_top_k: int = 100,
    return_rankings: bool = False,
) -> float | tuple[float, dict[int, list[int]]]:
    """Compute NDCG@k using precomputed query and markdown (dense) embeddings.

    Args:
        query_embeddings: List of query embedding tensors; same order as sorted query_ids from qrels.
        markdown_embeddings: List of corpus embedding tensors; index i = corpus position (corpus_id = i).
        qrels: Iterable of dicts with keys query_id, corpus_id, score (relevance 0/1/2).
        k: Rank cutoff for NDCG@k.
        reranker: Optional (tokenizer, model) tuple for cross-encoder reranking.
        query_texts: Raw query strings, same order as query_embeddings. Required when reranker is set.
        corpus_texts: Raw corpus strings, same order as markdown_embeddings. Required when reranker is set.
        rerank_top_k: Number of dense-retrieval candidates to rerank per query.
        return_rankings: If True, also return per-query top-k rankings as dict[query_id, list[corpus_id]].

    Returns:
        Mean NDCG@k in 0–100 scale, or (ndcg, rankings) if return_rankings is True.
    """
    ground_truth_pages = _relevants_from_qrels(qrels)
    query_ids = sorted(ground_truth_pages.keys())
    query_id_to_embedding = dict(zip(query_ids, query_embeddings))
    if reranker is not None:
        query_id_to_text = dict(zip(query_ids, query_texts))

    # Stack corpus embeddings (N, dim), L2-normalize for cosine = dot
    corpus = np.stack([_to_numpy(e) for e in markdown_embeddings], axis=0).astype(np.float32)
    norm = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus = corpus / (norm + 1e-9)

    ndcg_scores = []
    rankings: dict[int, list[int]] = {}
    for query_id in tqdm(ground_truth_pages, desc="NDCG@10"):
        q_emb = query_id_to_embedding[query_id]
        q = _to_numpy(q_emb)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        scores = (q @ corpus.T).flatten()

        if reranker is not None:
            candidate_positions = np.argsort(-scores)[: max(rerank_top_k, k)]
            candidate_texts_batch = [corpus_texts[int(p)] for p in candidate_positions]
            rerank_scores = rerank_candidates(
                reranker,
                query_id_to_text[query_id],
                candidate_texts_batch,
            )
            torch.cuda.empty_cache()
            reranked_order = np.argsort(-rerank_scores)
            top_k_corpus_ids = [int(candidate_positions[i]) for i in reranked_order[:k]]
        else:
            top_k_positions = np.argsort(-scores)[:k]
            top_k_corpus_ids = [int(p) for p in top_k_positions]

        rankings[query_id] = top_k_corpus_ids
        gt_pages = ground_truth_pages[query_id]
        relevance_at_rank = [gt_pages.get(cid, 0) for cid in top_k_corpus_ids]
        ndcg_scores.append(ndcg_at_k(relevance_at_rank, gt_pages, k=k))

    ndcg_at_10_mean = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    ndcg = ndcg_at_10_mean * 100
    return (ndcg, rankings) if return_rankings else ndcg
