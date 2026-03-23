import os
import tempfile

from tqdm import tqdm

from textual_extraction.config import CACHE_DIR_DEEPSEEK_MARKDOWNS, PROCESSED_DATA_DIR


def precompute_deepseek_markdowns(
    model,
    tokenizer,
    ds_corpus,
    save_dir: str = CACHE_DIR_DEEPSEEK_MARKDOWNS,
):
    """Run DeepSeek-OCR-2 on each corpus page image and save results to disk.

    Output layout per document: <save_dir>/<corpus_id>/result.mmd
    Already-processed documents are skipped (resume-safe).
    """
    save_dir = PROCESSED_DATA_DIR / save_dir
    os.makedirs(save_dir, exist_ok=True)

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    for corpus_id, image in tqdm(
        zip(ds_corpus["corpus_id"], ds_corpus["image"]),
        desc="Extracting markdown with DeepSeek-OCR-2",
        total=len(ds_corpus["corpus_id"]),
    ):
        output_path = save_dir / str(corpus_id)
        if (output_path / "result.mmd").exists():
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        try:
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_path,
                output_path=output_path,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )
        finally:
            os.unlink(temp_path)
