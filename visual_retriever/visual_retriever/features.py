import os

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from visual_retriever.config import PROCESSED_DATA_DIR
from visual_retriever.config import CACHE_DIR_IMAGE_EMBEDDINGS, CACHE_DIR_QUERY_EMBEDDINGS

app = typer.Typer()


def precompute_image_embeddings(
    model,
    ds_corpus,
    save_dir: str = CACHE_DIR_IMAGE_EMBEDDINGS,
    batch_size: int = 32,
    num_workers: int = 8,
):
    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    def collate_fn(batch):
        ids = [x["corpus_id"] for x in batch]
        imgs = [x["image"] for x in batch]  # PIL images
        return ids, imgs

    loader = DataLoader(
        ds_corpus,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model.eval()

    with torch.inference_mode():
        for ids, imgs in tqdm(loader):
            # Skip already cached
            todo = [
                (cid, img) for cid, img in zip(ids, imgs) if not (save_dir / f"{cid}.pt").exists()
            ]
            if not todo:
                continue

            todo_ids, todo_imgs = zip(*todo)
            embs = (
                model.forward_images(list(todo_imgs), batch_size=len(todo_imgs))
                .detach()
                .cpu()
                .to(torch.bfloat16)
            )

            for i, cid in enumerate(todo_ids):
                emb_i = embs[i].clone()  # or .contiguous()
                torch.save({"emb": emb_i}, save_dir / f"{cid}.pt")


def load_precomputed_image_embeddings(
    ds_corpus: Dataset, save_dir: str = CACHE_DIR_IMAGE_EMBEDDINGS
):
    # Load all the pages embeddings in memory

    save_dir = PROCESSED_DATA_DIR / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    pages_embeddings = [
        torch.load(save_dir / f"{cid}.pt", map_location="cpu")["emb"].to(torch.bfloat16)
        for cid in tqdm(ds_corpus["corpus_id"], desc="Loading pages embeddings")
    ]
    return pages_embeddings


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
        query_embedding = model.forward_queries([query]).to(torch.bfloat16)
        torch.save({"emb": query_embedding}, save_dir / f"{query_id}.pt")


def load_precomputed_query_embeddings(
    ds_queries: Dataset,
    save_dir: str = CACHE_DIR_QUERY_EMBEDDINGS,
):
    save_dir = PROCESSED_DATA_DIR / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    query_embeddings = [
        torch.load(save_dir / f"{query_id}.pt", map_location="cpu")["emb"].to(torch.bfloat16)
        for query_id in tqdm(ds_queries["query_id"], desc="Loading query embeddings")
    ]
    return query_embeddings
