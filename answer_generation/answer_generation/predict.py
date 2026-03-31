"""Generate answers for ViDoRe v3 queries using Ollama (qwen3.5:35b)."""

import json
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from answer_generation.config import (
    ALL_CONDITIONS,
    CONDITION_TO_MODALITY,
    GENERATOR_MODEL,
    MAX_CHARS_PER_DOC,
    PROCESSED_DATA_DIR,
    TOP_K,
    VIDORE_LANG,
    VIDORE_SUBSET,
    CONDITION_JINA_DEEPSEEK,
    CONDITION_JINA_DEEPSEEK_RERANKED,
    CONDITION_HYBRID_DEEPSEEK,
)
from answer_generation.dataset import load_data_vidore
from answer_generation.features import compute_top_k_per_query, load_deepseek_markdowns
from answer_generation.model import OllamaClient
from answer_generation.utils import (
    GENERATION_PROMPT_HYBRID,
    GENERATION_PROMPT_IMAGE,
    GENERATION_PROMPT_TEXT,
    format_documents,
)

app = typer.Typer()

_DEEPSEEK_CONDITIONS = {
    CONDITION_JINA_DEEPSEEK,
    CONDITION_JINA_DEEPSEEK_RERANKED,
    CONDITION_HYBRID_DEEPSEEK,
}


def _answers_dir(condition: str, subset: str) -> Path:
    return PROCESSED_DATA_DIR / "answers" / condition / subset


def _answer_path(condition: str, subset: str, query_id: int) -> Path:
    return _answers_dir(condition, subset) / f"{query_id}.json"


def _build_prompt(
    modality: str,
    query: str,
    markdown_texts: list[str],
) -> str:
    """Build generation prompt based on modality."""
    if modality == "text":
        return GENERATION_PROMPT_TEXT.format(
            documents=format_documents(markdown_texts),
            query=query,
        )
    if modality == "image":
        return GENERATION_PROMPT_IMAGE.format(query=query)
    if modality == "hybrid":
        return GENERATION_PROMPT_HYBRID.format(
            documents=format_documents(markdown_texts),
            query=query,
        )
    raise ValueError(f"Unknown modality: {modality!r}")


@app.command()
def main(
    subset: str = typer.Option(VIDORE_SUBSET, help="ViDoRe v3 subset name"),
    lang: str = typer.Option(VIDORE_LANG, help="Query language filter"),
    condition: str = typer.Option(
        ..., help=f"Retrieval condition. One of: {', '.join(ALL_CONDITIONS)}"
    ),
):
    """Generate answers for all queries under a given retrieval condition.

    Results are cached per query to data/processed/answers/<condition>/<query_id>.json.
    Already-cached queries are skipped (resume-safe).
    """
    if condition not in ALL_CONDITIONS:
        raise typer.BadParameter(
            f"Unknown condition {condition!r}. Valid: {', '.join(ALL_CONDITIONS)}"
        )

    modality = CONDITION_TO_MODALITY[condition]
    out_dir = _answers_dir(condition, subset)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset", subset=subset, lang=lang)
    ds_corpus, ds_queries, _ = load_data_vidore(subset, lang)

    # Build fast corpus lookups
    corpus_id_to_image = dict(zip(ds_corpus["corpus_id"], ds_corpus["image"]))
    corpus_id_to_nemo_markdown = dict(zip(ds_corpus["corpus_id"], ds_corpus["markdown"]))

    deepseek_markdowns: dict[int, str] = {}
    if condition in _DEEPSEEK_CONDITIONS:
        all_corpus_ids = list(ds_corpus["corpus_id"])
        deepseek_markdowns = load_deepseek_markdowns(subset, lang, all_corpus_ids)

    logger.info("Computing top-k rankings", condition=condition, top_k=TOP_K)
    rankings = compute_top_k_per_query(
        condition=condition,
        subset=subset,
        lang=lang,
        ds_queries=ds_queries,
        ds_corpus=ds_corpus,
        top_k=TOP_K,
    )

    client = OllamaClient()

    logger.info(f"Warming up model {GENERATOR_MODEL!r} (may take a while on cold start)...")
    client.chat(model=GENERATOR_MODEL, prompt="Hello")
    logger.info("Model ready.")

    query_id_to_text = dict(zip(ds_queries["query_id"], ds_queries["query"]))

    skipped = 0
    generated = 0

    for query_id in tqdm(ds_queries["query_id"], desc="Generating answers"):
        out_path = _answer_path(condition, subset, query_id)
        if out_path.exists():
            skipped += 1
            continue

        query = query_id_to_text[query_id]
        top_corpus_ids = rankings[query_id]

        # For hybrid conditions: visual top-k are the first TOP_K, textual top-k are the rest
        if modality == "hybrid":
            visual_ids = top_corpus_ids[:TOP_K]
            text_ids = top_corpus_ids[TOP_K:]
        else:
            visual_ids = top_corpus_ids if modality == "image" else []
            text_ids = top_corpus_ids if modality == "text" else []

        images = [corpus_id_to_image[cid] for cid in visual_ids] if visual_ids else None

        markdown_texts: list[str] = []
        if text_ids:
            if condition in _DEEPSEEK_CONDITIONS:
                raw_texts = [deepseek_markdowns.get(cid, "") for cid in text_ids]
            else:
                raw_texts = [corpus_id_to_nemo_markdown.get(cid, "") for cid in text_ids]

            for i, (cid, text) in enumerate(zip(text_ids, raw_texts)):
                if len(text) > MAX_CHARS_PER_DOC:
                    logger.debug(
                        "Document truncated",
                        query_id=query_id,
                        corpus_id=cid,
                        doc_index=i,
                        original_chars=len(text),
                        truncated_to=MAX_CHARS_PER_DOC,
                    )
                markdown_texts.append(text[:MAX_CHARS_PER_DOC])

        prompt = _build_prompt(modality, query, markdown_texts)

        doc_lengths = [len(t) for t in markdown_texts]
        logger.info(
            f"Sending to LLM | query_id={query_id} prompt_chars={len(prompt)} "
            f"num_images={len(images) if images else 0} doc_lengths={doc_lengths}"
        )

        answer = client.chat(
            model=GENERATOR_MODEL,
            prompt=prompt,
            images=images,
        )

        out_path.write_text(
            json.dumps(
                {
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "retrieved_corpus_ids": top_corpus_ids,
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
