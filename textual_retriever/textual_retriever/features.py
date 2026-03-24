import os

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from textual_retriever.config import PROCESSED_DATA_DIR
from textual_retriever.config import CACHE_DIR_QUERY_EMBEDDINGS, CACHE_DIR_MARKDOWN_EMBEDDINGS


def precompute_query_embeddings(
    model,
    ds_queries,
    save_dir: str = CACHE_DIR_QUERY_EMBEDDINGS,
):
    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    for query_id, query in tqdm(
        zip(ds_queries["query_id"], ds_queries["query"]), desc="Pre-computing query embeddings"
    ):
        if (save_dir / f"{query_id}.pt").exists():
            continue
        query_embedding = model.encode_text(
            texts=[query],
            prompt_name="query",
            task="retrieval",
        )[0]
        torch.save({"emb": query_embedding}, save_dir / f"{query_id}.pt")


def load_precomputed_query_embeddings(
    ds_queries: Dataset,
    save_dir: str = CACHE_DIR_QUERY_EMBEDDINGS,
):
    save_dir = PROCESSED_DATA_DIR / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    query_embeddings = [
        torch.load(save_dir / f"{query_id}.pt", map_location="cpu", weights_only=False)["emb"].to(
            torch.bfloat16
        )
        for query_id in tqdm(ds_queries["query_id"], desc="Loading query embeddings")
    ]
    return query_embeddings


def load_deepseek_markdowns_from_disk(ds_corpus, subset: str, lang: str) -> list[str]:
    """Load DeepSeek-OCR-2 extracted markdown from textual_extraction cache.

    Returns one string per corpus document (empty string if the file is missing).
    """
    from textual_retriever.config import PROJ_ROOT

    extraction_dir = (
        PROJ_ROOT.parent
        / "textual_extraction"
        / "data"
        / "processed"
        / f"deepseek_cache_markdowns_{subset}_{lang}"
    )
    return [
        (extraction_dir / str(corpus_id) / "result.mmd").read_text(encoding="utf-8")
        if (extraction_dir / str(corpus_id) / "result.mmd").exists()
        else ""
        for corpus_id in ds_corpus["corpus_id"]
    ]


def precompute_markdown_embeddings(
    model,
    ds_corpus,
    save_dir: str = CACHE_DIR_MARKDOWN_EMBEDDINGS,
    markdown_texts: list[str] | None = None,
):
    """Precompute Jina embeddings for corpus markdown texts.

    Args:
        markdown_texts: Override the markdown source. If None, uses ds_corpus["markdown"]
                        (NeMo/built-in). Pass the output of load_deepseek_markdowns_from_disk()
                        to embed DeepSeek-extracted text instead.
    """
    texts = markdown_texts if markdown_texts is not None else ds_corpus["markdown"]
    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    for corpus_id, markdown_text in tqdm(
        zip(ds_corpus["corpus_id"], texts),
        desc="Pre-computing markdown embeddings",
        total=len(ds_corpus["corpus_id"]),
    ):
        if (save_dir / f"{corpus_id}.pt").exists():
            continue
        emb = model.encode_text(
            texts=[markdown_text],
            prompt_name="passage",
            task="retrieval",
        )[0]

        torch.save({"emb": emb}, save_dir / f"{corpus_id}.pt")


def load_precomputed_markdown_embeddings(
    ds_corpus: Dataset,
    save_dir: str = CACHE_DIR_MARKDOWN_EMBEDDINGS,
):
    save_dir = PROCESSED_DATA_DIR / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    markdown_embeddings = []
    for corpus_id in tqdm(ds_corpus["corpus_id"], desc="Loading markdown embeddings"):
        emb = torch.load(save_dir / f"{corpus_id}.pt", map_location="cpu", weights_only=False)[
            "emb"
        ]
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        markdown_embeddings.append(emb.to(torch.float32))
    return markdown_embeddings
