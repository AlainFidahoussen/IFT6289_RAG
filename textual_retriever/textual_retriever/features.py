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


def precompute_markdown_embeddings(
    model,
    ds_corpus,
    save_dir: str = CACHE_DIR_MARKDOWN_EMBEDDINGS,
):
    """Precompute Jina embeddings for corpus markdown texts. Uses corpus_id and markdown columns."""
    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    for corpus_id, markdown_text in tqdm(
        zip(ds_corpus["corpus_id"], ds_corpus["markdown"]),
        desc="Pre-computing markdown embeddings",
        total=len(ds_corpus["corpus_id"]),
    ):
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
