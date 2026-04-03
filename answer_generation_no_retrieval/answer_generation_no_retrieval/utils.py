"""Prompt templates, result types, and metrics for closed-book generation and judging."""

import json
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_PROMPT_CLOSED_BOOK = """\
Answer the following query: {query}

Keep the response short when appropriate. Output the answer only."""


JUDGE_PROMPT = """\
You are an expert evaluator assessing whether a generated answer is correct.

Question: {query}
Ground-truth answer: {true_answer}
Generated answer: {test_answer}

Judge whether the generated answer is correct. An answer is correct if it captures \
the essential information from the ground-truth answer, even if phrased differently. \
Partial or vague answers should be marked Incorrect.

Respond with a JSON object with exactly two fields:
- "judgment": either "Correct" or "Incorrect"
- "explanation": one sentence explaining your decision

JSON:"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class JudgmentResult:
    query_id: int
    judgment: str  # "Correct" | "Incorrect"
    explanation: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_judge_response(query_id: int, raw: str) -> JudgmentResult:
    """Parse a JSON judge response into a JudgmentResult.

    Falls back to regex extraction if the model returns malformed JSON,
    as long as "Correct" or "Incorrect" is unambiguously present.

    Raises:
        ValueError: if neither JSON parsing nor regex can extract a valid judgment.
    """
    try:
        data = json.loads(raw)
        judgment = data.get("judgment", "")
        if judgment in ("Correct", "Incorrect"):
            return JudgmentResult(
                query_id=query_id,
                judgment=judgment,
                explanation=data.get("explanation", ""),
            )
        if judgment:
            return JudgmentResult(
                query_id=query_id,
                judgment="Incorrect",
                explanation=data.get("explanation", f"Non-binary judgment normalized: {judgment}"),
            )
    except json.JSONDecodeError:
        pass

    match = re.search(r'"judgment"\s*:\s*"(Correct|Incorrect)"', raw)
    if match:
        judgment = match.group(1)
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', raw)
        explanation = explanation_match.group(1) if explanation_match else ""
        return JudgmentResult(query_id=query_id, judgment=judgment, explanation=explanation)

    raise ValueError(
        f"Could not extract judgment for query_id={query_id}. Raw response: {raw!r}"
    )


def compute_pass_at_1(judgments: list[JudgmentResult]) -> float:
    """Fraction of judgments labeled 'Correct' (pass@1)."""
    if not judgments:
        return 0.0
    return sum(1 for j in judgments if j.judgment == "Correct") / len(judgments)
