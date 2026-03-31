"""Compute top-k corpus IDs per query from precomputed retrieval embeddings."""

import json
import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from answer_generation.config import (
    DEEPSEEK_EXTRACTION_DIR,
    RERANK_TOP_K,
    REPO_ROOT,
    TEXTUAL_CACHE_DIR,
    VISUAL_CACHE_DIR,
    CONDITION_COLEMBED,
    CONDITION_HYBRID_DEEPSEEK,
    CONDITION_HYBRID_NEMO,
    CONDITION_JINA_DEEPSEEK,
    CONDITION_JINA_DEEPSEEK_RERANKED,
    CONDITION_JINA_NEMO,
    CONDITION_JINA_NEMO_RERANKED,
)

# --- Embedding loaders ---


def _load_embeddings_from_dir(cache_dir, ids: list[int]) -> list[torch.Tensor]:
    """Load .pt embeddings for given IDs from a cache directory."""
    embeddings: list[torch.Tensor] = []
    for item_id in ids:
        path = cache_dir / f"{item_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Embedding not found: {path}")
        emb = torch.load(path, map_location="cpu", weights_only=False)["emb"]
        embeddings.append(emb)
    return embeddings


def _sorted_ids_from_dir(cache_dir) -> list[int]:
    """Return corpus/query IDs from a cache dir, sorted numerically."""
    return sorted(int(p.stem) for p in cache_dir.glob("*.pt"))


def load_textual_corpus_embeddings(
    subset: str, lang: str, source: str
) -> tuple[list[int], list[torch.Tensor]]:
    """Load Jina v4 corpus embeddings (nemo or deepseek source).

    Returns:
        (corpus_ids, embeddings) — parallel lists in sorted corpus_id order.
    """
    if source == "deepseek":
        cache_dir = TEXTUAL_CACHE_DIR / f"jina_cache_markdowns_deepseek_{subset}_{lang}"
    else:
        cache_dir = TEXTUAL_CACHE_DIR / f"jina_cache_markdowns_{subset}_{lang}"
    corpus_ids = _sorted_ids_from_dir(cache_dir)
    return corpus_ids, _load_embeddings_from_dir(cache_dir, corpus_ids)


def load_textual_query_embeddings(
    subset: str, lang: str, query_ids: list[int]
) -> dict[int, torch.Tensor]:
    """Load Jina v4 query embeddings keyed by query_id."""
    cache_dir = TEXTUAL_CACHE_DIR / f"jina_cache_queries_{subset}_{lang}"
    embs = _load_embeddings_from_dir(cache_dir, query_ids)
    return dict(zip(query_ids, embs))


def load_visual_corpus_embeddings(
    subset: str, lang: str
) -> tuple[list[int], list[torch.Tensor]]:
    """Load ColEmbed corpus image embeddings.

    Returns:
        (corpus_ids, embeddings) — parallel lists in sorted corpus_id order.
    """
    cache_dir = VISUAL_CACHE_DIR / f"colembed_cache_pages_{subset}_{lang}"
    corpus_ids = _sorted_ids_from_dir(cache_dir)
    return corpus_ids, _load_embeddings_from_dir(cache_dir, corpus_ids)


def load_visual_query_embeddings(
    subset: str, lang: str, query_ids: list[int]
) -> dict[int, torch.Tensor]:
    """Load ColEmbed query embeddings keyed by query_id."""
    cache_dir = VISUAL_CACHE_DIR / f"colembed_cache_queries_{subset}_{lang}"
    embs = _load_embeddings_from_dir(cache_dir, query_ids)
    return dict(zip(query_ids, embs))


def load_deepseek_markdowns(
    subset: str, lang: str, corpus_ids: list[int]
) -> dict[int, str]:
    """Load DeepSeek-OCR-2 markdown texts for given corpus IDs."""
    base = DEEPSEEK_EXTRACTION_DIR / f"deepseek_cache_markdowns_{subset}_{lang}"
    result: dict[int, str] = {}
    for cid in corpus_ids:
        path = base / str(cid) / "result.mmd"
        result[cid] = path.read_text(encoding="utf-8") if path.exists() else ""
    return result


# --- Scoring functions ---


def _textual_cosine_top_k(
    query_emb: torch.Tensor,
    corpus_ids: list[int],
    corpus_matrix: np.ndarray,
    k: int,
) -> list[int]:
    """Top-k corpus IDs by cosine similarity against a prebuilt L2-normalized matrix."""
    q = query_emb.float().numpy()
    if q.ndim == 2:
        q = q[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = corpus_matrix @ q  # [N]
    top_positions = np.argsort(-scores)[:k]
    return [corpus_ids[int(i)] for i in top_positions]


def _build_textual_matrix(corpus_embs: list[torch.Tensor]) -> np.ndarray:
    """Stack and L2-normalize corpus embeddings into a scoring matrix [N, d]."""
    matrix = np.stack([e.float().numpy() for e in corpus_embs], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-9)


def _visual_maxsim_top_k(
    query_emb: torch.Tensor,
    corpus_ids: list[int],
    corpus_embs: list[torch.Tensor],
    k: int,
) -> list[int]:
    """Top-k corpus IDs by ColEmbed MaxSim (no model required).

    MaxSim: for each query token, take max cosine sim over all page tokens, then sum.
    """
    q = query_emb.float()
    if q.ndim == 3:
        q = q.squeeze(0)  # [m, d]

    scores = np.empty(len(corpus_embs), dtype=np.float32)
    for i, page_emb in enumerate(corpus_embs):
        p = page_emb.float()  # [T, d]
        sim = q @ p.T  # [m, T]
        scores[i] = sim.max(dim=1).values.sum().item()

    top_positions = np.argsort(-scores)[:k]
    return [corpus_ids[int(i)] for i in top_positions]


# --- Cached rankings loader ---

_TEXTUAL_RETRIEVER_DIR = REPO_ROOT / "textual_retriever"
_VISUAL_RETRIEVER_DIR = REPO_ROOT / "visual_retriever"


def _load_cached_rankings(path) -> dict[int, list[int]] | None:
    """Load a rankings JSON file. Returns None if the file does not exist."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _load_cached_textual_rankings(
    condition: str, subset: str, lang: str
) -> dict[int, list[int]] | None:
    """Load pre-saved textual rankings from textual_retriever if available.

    Produced by: cd textual_retriever && bash run_all.sh [--rerank] [--deepseek]
    """
    path = _TEXTUAL_RETRIEVER_DIR / f"data/processed/rankings_{condition}_{subset}_{lang}.json"
    return _load_cached_rankings(path)


def _load_cached_visual_rankings(
    subset: str, lang: str
) -> dict[int, list[int]] | None:
    """Load pre-saved ColEmbed rankings from visual_retriever if available.

    Produced by: cd visual_retriever && bash run_all.sh
    """
    path = _VISUAL_RETRIEVER_DIR / f"data/processed/rankings_colembed_{subset}_{lang}.json"
    return _load_cached_rankings(path)


# --- Main entry point ---


def compute_top_k_per_query(
    condition: str,
    subset: str,
    lang: str,
    ds_queries: Dataset,
    ds_corpus: Dataset,
    top_k: int,
) -> dict[int, list[int]]:
    """Compute top-k retrieved corpus IDs per query for a given retrieval condition.

    Args:
        condition: One of the CONDITION_* constants from config.
        ds_corpus: Full ViDoRe corpus dataset (needed for NeMo markdown texts when reranking).
        top_k: Number of pages to retrieve per query.

    Returns:
        Dict mapping query_id → list of top-k corpus_ids.
    """
    query_ids = list(ds_queries["query_id"])

    # Fast path: load pre-saved textual rankings (avoids embedding load + reranking overhead)
    cached = _load_cached_textual_rankings(condition, subset, lang)
    if cached is not None and condition not in (CONDITION_HYBRID_NEMO, CONDITION_HYBRID_DEEPSEEK):
        logger.info(f"Loaded cached textual rankings for {condition} ({subset})")
        return {qid: cached[qid][:top_k] for qid in query_ids if qid in cached}

    # Fast path: load pre-saved ColEmbed rankings
    if condition == CONDITION_COLEMBED:
        cached_vis = _load_cached_visual_rankings(subset, lang)
        if cached_vis is not None:
            logger.info(f"Loaded cached visual rankings for colembed ({subset})")
            return {qid: cached_vis[qid][:top_k] for qid in query_ids if qid in cached_vis}

    # For hybrid conditions: load cached reranked textual + visual rankings (paper protocol)
    cached_text_rankings: dict[int, list[int]] = {}
    cached_vis_rankings: dict[int, list[int]] = {}
    if condition in (CONDITION_HYBRID_NEMO, CONDITION_HYBRID_DEEPSEEK):
        reranked_condition = (
            CONDITION_JINA_DEEPSEEK_RERANKED
            if "deepseek" in condition
            else CONDITION_JINA_NEMO_RERANKED
        )
        cached_text_rankings = _load_cached_textual_rankings(reranked_condition, subset, lang) or {}
        if cached_text_rankings:
            logger.info(f"Using cached reranked textual rankings for hybrid ({reranked_condition})")
        else:
            logger.warning(
                f"No cached reranked rankings found for {reranked_condition} ({subset}). "
                "Run: cd textual_retriever && bash run_all.sh --rerank [--deepseek]"
            )
        cached_vis_rankings = _load_cached_visual_rankings(subset, lang) or {}
        if cached_vis_rankings:
            logger.info(f"Using cached visual rankings for hybrid ({subset})")
        else:
            logger.warning(
                f"No cached visual rankings found for ({subset}). "
                "Run: cd visual_retriever && bash run_all.sh"
            )

    # For hybrid: if both caches are available, no embeddings need to be loaded at all
    hybrid_fully_cached = (
        condition in (CONDITION_HYBRID_NEMO, CONDITION_HYBRID_DEEPSEEK)
        and bool(cached_text_rankings)
        and bool(cached_vis_rankings)
    )

    needs_textual = not hybrid_fully_cached and condition in (
        CONDITION_JINA_NEMO,
        CONDITION_JINA_NEMO_RERANKED,
        CONDITION_JINA_DEEPSEEK,
        CONDITION_JINA_DEEPSEEK_RERANKED,
        CONDITION_HYBRID_NEMO,
        CONDITION_HYBRID_DEEPSEEK,
    )
    needs_visual = not hybrid_fully_cached and condition in (
        CONDITION_COLEMBED,
        CONDITION_HYBRID_NEMO,
        CONDITION_HYBRID_DEEPSEEK,
    )
    needs_rerank = condition in (CONDITION_JINA_NEMO_RERANKED, CONDITION_JINA_DEEPSEEK_RERANKED)

    text_source = "deepseek" if "deepseek" in condition else "nemo"

    if needs_textual:
        logger.info("Loading textual corpus embeddings", condition=condition, source=text_source)
        text_corpus_ids, text_corpus_embs = load_textual_corpus_embeddings(
            subset, lang, text_source
        )
        text_matrix = _build_textual_matrix(text_corpus_embs)
        logger.info("Loading textual query embeddings", subset=subset)
        text_query_emb_map = load_textual_query_embeddings(subset, lang, query_ids)

    if needs_visual:
        logger.info("Loading visual corpus embeddings", condition=condition)
        vis_corpus_ids, vis_corpus_embs = load_visual_corpus_embeddings(subset, lang)
        logger.info("Loading visual query embeddings", subset=subset)
        vis_query_emb_map = load_visual_query_embeddings(subset, lang, query_ids)

    reranker = None
    query_id_to_text: dict[int, str] = {}
    corpus_id_to_text: dict[int, str] = {}
    if needs_rerank:
        from answer_generation.model import load_zerank2

        logger.info("Loading zerank-2 reranker for condition", condition=condition)
        reranker = load_zerank2()
        query_id_to_text = dict(zip(ds_queries["query_id"], ds_queries["query"]))
        if text_source == "deepseek":
            corpus_id_to_text = load_deepseek_markdowns(subset, lang, text_corpus_ids)
        else:
            corpus_id_to_text = dict(zip(ds_corpus["corpus_id"], ds_corpus["markdown"]))

    rankings: dict[int, list[int]] = {}

    for query_id in tqdm(query_ids, desc=f"Retrieving top-{top_k} ({condition})"):
        if condition == CONDITION_COLEMBED:
            q_emb = vis_query_emb_map[query_id]
            rankings[query_id] = _visual_maxsim_top_k(q_emb, vis_corpus_ids, vis_corpus_embs, top_k)

        elif condition in (CONDITION_JINA_NEMO, CONDITION_JINA_DEEPSEEK):
            q_emb = text_query_emb_map[query_id]
            rankings[query_id] = _textual_cosine_top_k(q_emb, text_corpus_ids, text_matrix, top_k)

        elif needs_rerank:
            q_emb = text_query_emb_map[query_id]
            # Dense top-RERANK_TOP_K candidates, then rerank
            candidates = _textual_cosine_top_k(
                q_emb, text_corpus_ids, text_matrix, max(RERANK_TOP_K, top_k)
            )
            candidate_texts = [corpus_id_to_text.get(cid, "") for cid in candidates]
            pairs = [[query_id_to_text[query_id], t] for t in candidate_texts]
            rerank_scores = reranker.predict(pairs, batch_size=1, show_progress_bar=False)
            reranked_order = np.argsort(-np.asarray(rerank_scores, dtype=np.float32))[:top_k]
            rankings[query_id] = [candidates[int(i)] for i in reranked_order]

        elif condition in (CONDITION_HYBRID_NEMO, CONDITION_HYBRID_DEEPSEEK):
            # Paper protocol: visual top-k + reranked textual top-k (no dedup)
            if query_id in cached_vis_rankings:
                vis_top = cached_vis_rankings[query_id][:top_k]
            else:
                q_vis = vis_query_emb_map[query_id]
                vis_top = _visual_maxsim_top_k(q_vis, vis_corpus_ids, vis_corpus_embs, top_k)

            if query_id in cached_text_rankings:
                txt_top = cached_text_rankings[query_id][:top_k]
            else:
                q_txt = text_query_emb_map[query_id]
                txt_top = _textual_cosine_top_k(q_txt, text_corpus_ids, text_matrix, top_k)

            rankings[query_id] = vis_top + txt_top

        else:
            raise ValueError(f"Unknown condition: {condition!r}")

    return rankings
