"""Generate answers for ViDoRe v3 queries without retrieval (closed-book ablation)."""

import json
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from answer_generation_no_retrieval.config import (
    CONDITION_CLOSED_BOOK,
    GENERATOR_MODEL,
    PROCESSED_DATA_DIR,
    VIDORE_LANG,
    VIDORE_SUBSET,
)
from answer_generation_no_retrieval.dataset import load_data_vidore
from answer_generation_no_retrieval.model import OllamaClient
from answer_generation_no_retrieval.utils import GENERATION_PROMPT_CLOSED_BOOK

app = typer.Typer()


def _answers_dir(condition: str, subset: str) -> Path:
    return PROCESSED_DATA_DIR / "answers" / condition / subset


def _answer_path(condition: str, subset: str, query_id: int) -> Path:
    return _answers_dir(condition, subset) / f"{query_id}.json"


@app.command()
def main(
    subset: str = typer.Option(VIDORE_SUBSET, help="ViDoRe v3 subset name"),
    lang: str = typer.Option(VIDORE_LANG, help="Query language filter"),
):
    """Generate closed-book answers for all queries (no retrieval).

    Results are cached per query to data/processed/answers/closed_book/<subset>/<query_id>.json.
    Already-cached queries are skipped (resume-safe).
    """
    condition = CONDITION_CLOSED_BOOK
    out_dir = _answers_dir(condition, subset)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset", subset=subset, lang=lang)
    _, ds_queries, _ = load_data_vidore(subset, lang)

    client = OllamaClient()

    logger.info(f"Warming up model {GENERATOR_MODEL!r} (may take a while on cold start)...")
    client.chat(model=GENERATOR_MODEL, prompt="Hello")
    logger.info("Model ready.")

    query_id_to_text = dict(zip(ds_queries["query_id"], ds_queries["query"]))

    skipped = 0
    generated = 0

    for query_id in tqdm(ds_queries["query_id"], desc="Generating answers (closed-book)"):
        out_path = _answer_path(condition, subset, query_id)
        if out_path.exists():
            skipped += 1
            continue

        query = query_id_to_text[query_id]
        prompt = GENERATION_PROMPT_CLOSED_BOOK.format(query=query)

        logger.info(
            f"Sending to LLM | query_id={query_id} prompt_chars={len(prompt)} "
            f"num_images=0 num_docs=0"
        )

        answer = client.chat(
            model=GENERATOR_MODEL,
            prompt=prompt,
        )

        out_path.write_text(
            json.dumps(
                {
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "retrieved_corpus_ids": [],
                    "condition": condition,
                    "subset": subset,
                    "lang": lang,
                    "generator_model": GENERATOR_MODEL,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        generated += 1

    logger.success(
        "Answer generation complete",
        subset=subset,
        condition=condition,
        generated=generated,
        skipped=skipped,
    )


if __name__ == "__main__":
    app()
