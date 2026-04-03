"""LLM-as-judge: score closed-book answers with llama3.1:8b via Ollama."""

import csv
import json
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from answer_generation_no_retrieval.config import (
    CONDITION_CLOSED_BOOK,
    JUDGE_MODEL,
    PROCESSED_DATA_DIR,
    VIDORE_LANG,
    VIDORE_SUBSET,
)
from answer_generation_no_retrieval.dataset import get_answer_field, load_data_vidore
from answer_generation_no_retrieval.model import OllamaClient
from answer_generation_no_retrieval.utils import (
    JUDGE_PROMPT,
    JudgmentResult,
    compute_pass_at_1,
    parse_judge_response,
)

app = typer.Typer()

RESULTS_FILE = Path("results_answers.csv")


def _judgments_dir(condition: str, subset: str) -> Path:
    return PROCESSED_DATA_DIR / "judgments" / condition / subset


def _judgment_path(condition: str, subset: str, query_id: int) -> Path:
    return _judgments_dir(condition, subset) / f"{query_id}.json"


def _answers_dir(condition: str, subset: str) -> Path:
    return PROCESSED_DATA_DIR / "answers" / condition / subset


def _load_generated_answer(condition: str, subset: str, query_id: int) -> str | None:
    """Load a previously generated answer, or None if it does not exist."""
    path = _answers_dir(condition, subset) / f"{query_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["answer"]


@app.command()
def main(
    subset: str = typer.Option(VIDORE_SUBSET, help="ViDoRe v3 subset name"),
    lang: str = typer.Option(VIDORE_LANG, help="Query language filter"),
):
    """Judge closed-book answers and compute pass@1.

    Reads generated answers from data/processed/answers/closed_book/.
    Saves per-query judgments to data/processed/judgments/closed_book/.
    Appends pass@1 to results_answers.csv.
    """
    condition = CONDITION_CLOSED_BOOK
    out_dir = _judgments_dir(condition, subset)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset", subset=subset, lang=lang)
    _, ds_queries, _ = load_data_vidore(subset, lang)

    answer_field = get_answer_field(ds_queries)
    logger.info("Ground-truth answer field detected", field=answer_field)

    query_id_to_text = dict(zip(ds_queries["query_id"], ds_queries["query"]))
    query_id_to_true_answer = dict(zip(ds_queries["query_id"], ds_queries[answer_field]))

    client = OllamaClient()
    judgments: list[JudgmentResult] = []
    skipped = 0
    missing = 0

    for query_id in tqdm(ds_queries["query_id"], desc="Judging answers"):
        judgment_path = _judgment_path(condition, subset, query_id)

        if judgment_path.exists():
            cached = json.loads(judgment_path.read_text(encoding="utf-8"))
            judgments.append(
                JudgmentResult(
                    query_id=query_id,
                    judgment=cached["judgment"],
                    explanation=cached["explanation"],
                )
            )
            skipped += 1
            continue

        generated_answer = _load_generated_answer(condition, subset, query_id)
        if generated_answer is None:
            logger.warning(
                "No generated answer found, skipping",
                query_id=query_id,
                condition=condition,
            )
            missing += 1
            continue

        prompt = JUDGE_PROMPT.format(
            query=query_id_to_text[query_id],
            true_answer=query_id_to_true_answer[query_id],
            test_answer=generated_answer,
        )

        raw_response = client.chat(
            model=JUDGE_MODEL,
            prompt=prompt,
            response_format="json",
        )

        result = parse_judge_response(query_id, raw_response)
        judgments.append(result)

        judgment_path.write_text(
            json.dumps(
                {
                    "query_id": query_id,
                    "query": query_id_to_text[query_id],
                    "gt_answer": query_id_to_true_answer[query_id],
                    "predicted_answer": generated_answer,
                    "judgment": result.judgment,
                    "explanation": result.explanation,
                    "condition": condition,
                    "subset": subset,
                    "lang": lang,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not judgments:
        logger.error(
            "No judgments produced", condition=condition, subset=subset, missing=missing
        )
        raise typer.Exit(code=1)

    pass_at_1 = compute_pass_at_1(judgments)
    num_correct = sum(1 for j in judgments if j.judgment == "Correct")

    logger.success(
        f"Judging complete | subset={subset} condition={condition} "
        f"pass@1={pass_at_1:.3f} ({num_correct}/{len(judgments)}) "
        f"skipped={skipped} missing={missing}"
    )

    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["timestamp", "condition", "subset", "lang", "pass_at_1", "num_correct", "num_total"]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                condition,
                subset,
                lang,
                f"{pass_at_1:.4f}",
                num_correct,
                len(judgments),
            ]
        )
    logger.info(f"Result appended to {RESULTS_FILE}")


if __name__ == "__main__":
    app()
