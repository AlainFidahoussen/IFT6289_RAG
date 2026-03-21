import os
from datasets import Dataset
from tqdm import tqdm
import torch
from textual_extraction.config import CACHE_DIR_MARKDOWN_EMBEDDINGS
from textual_extraction.config import PROCESSED_DATA_DIR
import tempfile

def precompute_markdown_embeddings(
    model,
    tokenizer,
    ds_corpus,
    save_dir: str = CACHE_DIR_MARKDOWN_EMBEDDINGS,
):

    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    for corpus_id, image in tqdm(
        zip(ds_corpus["corpus_id"], ds_corpus["image"]),
        desc="Pre-computing markdown embeddings",
        total=len(ds_corpus["corpus_id"]),
    ):
        output_path = save_dir / f"{corpus_id}"
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            image_file = temp_file.name
            response = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = True)


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
        markdown_embeddings.append(emb)

    return markdown_embeddings
