"""Shared loaders for post-hoc analysis.

Reads cached artifacts produced by other subprojects:
  answer_generation/data/processed/{answers,judgments}/<condition>/<subset>/<query_id>.json
  answer_generation_no_retrieval/data/processed/{answers,judgments}/closed_book/<subset>/<query_id>.json
  textual_retriever/data/processed/rankings_{condition}_{subset}_{lang}.json
  visual_retriever/data/processed/rankings_colembed_{subset}_{lang}.json
  textual_extraction/data/processed/deepseek_cache_markdowns_{subset}_{lang}/<corpus_id>/result.mmd

And the upstream ViDoRe v3 HuggingFace dataset (via the existing loader in answer_generation).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
ANSWER_GEN_DIR: Path = REPO_ROOT / "answer_generation"
ANSWER_GEN_NO_RETRIEVAL_DIR: Path = REPO_ROOT / "answer_generation_no_retrieval"
TEXTUAL_RETRIEVER_DIR: Path = REPO_ROOT / "textual_retriever"
VISUAL_RETRIEVER_DIR: Path = REPO_ROOT / "visual_retriever"
TEXTUAL_EXTRACTION_DIR: Path = REPO_ROOT / "textual_extraction"

# Subset → query-filter language (the language the queries are in, not the corpus).
# For every subset we only ever evaluated the "native" language (EN on EN corpora,
# FR on FR corpora), so this is a straight map.
SUBSET_LANG: dict[str, str] = {
    "computer_science": "english",
    "finance_en": "english",
    "pharmaceuticals": "english",
    "physics": "french",
    "finance_fr": "french",
}

RETRIEVAL_CONDITIONS: tuple[str, ...] = (
    "jina_nemo",
    "jina_nemo_reranked",
    "jina_deepseek",
    "jina_deepseek_reranked",
    "colembed",
    "hybrid_nemo",
    "hybrid_deepseek",
)

CLOSED_BOOK_CONDITION: str = "closed_book"


# -----------------------------------------------------------------------------
# Judgments + answers
# -----------------------------------------------------------------------------


def _judgments_root(condition: str) -> Path:
    """Where the per-query judgment JSONs live for a given condition."""
    if condition == CLOSED_BOOK_CONDITION:
        return ANSWER_GEN_NO_RETRIEVAL_DIR / "data" / "processed" / "judgments" / condition
    return ANSWER_GEN_DIR / "data" / "processed" / "judgments" / condition


def _answers_root(condition: str) -> Path:
    if condition == CLOSED_BOOK_CONDITION:
        return ANSWER_GEN_NO_RETRIEVAL_DIR / "data" / "processed" / "answers" / condition
    return ANSWER_GEN_DIR / "data" / "processed" / "answers" / condition


def _load_one_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_judgments(
    subsets: list[str],
    conditions: list[str],
) -> pd.DataFrame:
    """Walk cached judgment JSONs and return a long-form DataFrame.

    Columns:
        query_id, condition, subset, lang, judgment ("Correct" | "Incorrect"),
        explanation, predicted_answer, gt_answer, query.
    """
    rows: list[dict] = []
    for condition in conditions:
        root = _judgments_root(condition)
        for subset in subsets:
            subset_dir = root / subset
            if not subset_dir.is_dir():
                logger.warning("No judgments dir", condition=condition, subset=subset, path=str(subset_dir))
                continue
            for path in subset_dir.glob("*.json"):
                data = _load_one_json(path)
                rows.append(
                    {
                        "query_id": int(data["query_id"]),
                        "condition": condition,
                        "subset": subset,
                        "lang": data.get("lang", SUBSET_LANG[subset]),
                        "judgment": data["judgment"],
                        "explanation": data.get("explanation", ""),
                        "predicted_answer": data.get("predicted_answer", ""),
                        "gt_answer": data.get("gt_answer", ""),
                        "query": data.get("query", ""),
                    }
                )
    df = pd.DataFrame(rows)
    df["correct"] = (df["judgment"] == "Correct").astype(int)
    return df


def load_answers(
    subsets: list[str],
    conditions: list[str],
) -> pd.DataFrame:
    """Walk cached answer JSONs and return a long-form DataFrame.

    Columns:
        query_id, condition, subset, lang, answer, retrieved_corpus_ids.
    """
    rows: list[dict] = []
    for condition in conditions:
        root = _answers_root(condition)
        for subset in subsets:
            subset_dir = root / subset
            if not subset_dir.is_dir():
                logger.warning("No answers dir", condition=condition, subset=subset, path=str(subset_dir))
                continue
            for path in subset_dir.glob("*.json"):
                data = _load_one_json(path)
                rows.append(
                    {
                        "query_id": int(data["query_id"]),
                        "condition": condition,
                        "subset": subset,
                        "lang": data.get("lang", SUBSET_LANG[subset]),
                        "answer": data.get("answer", ""),
                        "retrieved_corpus_ids": tuple(data.get("retrieved_corpus_ids", [])),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Rankings
# -----------------------------------------------------------------------------


def _ranking_path(condition: str, subset: str, lang: str) -> Path:
    if condition == "colembed":
        return VISUAL_RETRIEVER_DIR / "data" / "processed" / f"rankings_colembed_{subset}_{lang}.json"
    return (
        TEXTUAL_RETRIEVER_DIR
        / "data"
        / "processed"
        / f"rankings_{condition}_{subset}_{lang}.json"
    )


def load_rankings(condition: str, subset: str, lang: str | None = None) -> dict[int, list[int]]:
    """Load cached top-k rankings for a ranking-producing condition.

    Returns: dict mapping query_id (int) → list of corpus_ids.
    """
    if lang is None:
        lang = SUBSET_LANG[subset]
    path = _ranking_path(condition, subset, lang)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


# -----------------------------------------------------------------------------
# ViDoRe v3 query metadata
# -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_query_metadata(subset: str) -> pd.DataFrame:
    """Load ViDoRe v3 query metadata for the language we actually evaluated.

    Returns one row per query_id with columns:
        query_id, subset, lang, query, query_types (list[str]), content_type (list[str]),
        query_format, source_type, query_type_for_generation.

    The list-valued columns are kept as Python lists so callers can explode as they need.
    """
    from datasets import load_dataset

    lang = SUBSET_LANG[subset]
    ds = load_dataset(f"vidore/vidore_v3_{subset}", "queries", split="test")
    ds = ds.filter(lambda x: x["language"] == lang)

    df = pd.DataFrame(
        {
            "query_id": [int(q) for q in ds["query_id"]],
            "subset": [subset] * len(ds),
            "lang": [lang] * len(ds),
            "query": ds["query"],
            "query_types": list(ds["query_types"]),
            "content_type": list(ds["content_type"]),
            "query_format": ds["query_format"],
            "source_type": ds["source_type"],
            "query_type_for_generation": ds["query_type_for_generation"],
        }
    )
    # First element of each list → a "primary" categorical for easy grouping.
    df["query_types_primary"] = df["query_types"].apply(
        lambda xs: xs[0] if isinstance(xs, list) and xs else "unknown"
    )
    df["content_type_primary"] = df["content_type"].apply(
        lambda xs: xs[0] if isinstance(xs, list) and xs else "unknown"
    )
    return df


def load_all_query_metadata(subsets: list[str]) -> pd.DataFrame:
    frames = [load_query_metadata(s) for s in subsets]
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Parser output (markdown text)
# -----------------------------------------------------------------------------


def load_deepseek_markdown(subset: str, corpus_id: int) -> str:
    """Read a single DeepSeek-OCR-2 markdown file. Returns '' if missing."""
    lang = SUBSET_LANG[subset]
    path = (
        TEXTUAL_EXTRACTION_DIR
        / "data"
        / "processed"
        / f"deepseek_cache_markdowns_{subset}_{lang}"
        / str(corpus_id)
        / "result.mmd"
    )
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_nemo_markdown_map(subset: str) -> dict[int, str]:
    """Read NeMo markdown for every corpus page from the HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(f"vidore/vidore_v3_{subset}", "corpus", split="test")
    return {int(c): (m or "") for c, m in zip(ds["corpus_id"], ds["markdown"])}


# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------


RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "results"


def results_path(filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename


__all__ = [
    "SUBSET_LANG",
    "RETRIEVAL_CONDITIONS",
    "CLOSED_BOOK_CONDITION",
    "load_judgments",
    "load_answers",
    "load_rankings",
    "load_query_metadata",
    "load_all_query_metadata",
    "load_deepseek_markdown",
    "load_nemo_markdown_map",
    "results_path",
]
