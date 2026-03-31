"""Print generation and judging progress across all conditions and subsets."""

import json
from pathlib import Path

PROCESSED = Path(__file__).parent / "data" / "processed"
ANSWERS_DIR = PROCESSED / "answers"
JUDGMENTS_DIR = PROCESSED / "judgments"

TOTAL_QUERIES = {
    "computer_science": 215,
    "finance_en": 309,
    "pharmaceuticals": 364,
}
SUBSETS = list(TOTAL_QUERIES)
CONDITIONS = [
    "jina_nemo",
    "jina_nemo_reranked",
    "jina_deepseek",
    "jina_deepseek_reranked",
    "colembed",
    "hybrid_nemo",
    "hybrid_deepseek",
]


def count_by_subset(directory: Path) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in SUBSETS}
    if not directory.exists():
        return counts
    for subset in SUBSETS:
        counts[subset] = len(list((directory / subset).glob("*.json"))) if (directory / subset).exists() else 0
    return counts


def fmt(done: int, total: int) -> str:
    pct = done * 100 // total
    return f"{done:>3}/{total} ({pct:>3}%)"


def print_table(label: str, base_dir: Path) -> None:
    col_w = max(len(s) for s in SUBSETS) + 2
    cond_w = max(len(c) for c in CONDITIONS) + 2

    header = f"{'Condition':<{cond_w}}" + "".join(f"{s:>{col_w}}" for s in SUBSETS) + f"{'Total':>{col_w}}"
    print(f"\n{'─' * len(header)}")
    print(f"  {label}")
    print(f"{'─' * len(header)}")
    print(header)
    print("─" * len(header))

    for cond in CONDITIONS:
        counts = count_by_subset(base_dir / cond)
        total_done = sum(counts.values())
        total_expected = sum(TOTAL_QUERIES.values())
        row = f"{cond:<{cond_w}}"
        row += "".join(f"{fmt(counts[s], TOTAL_QUERIES[s]):>{col_w}}" for s in SUBSETS)
        row += f"{fmt(total_done, total_expected):>{col_w}}"
        print(row)

    print("─" * len(header))


print_table("ANSWER GENERATION progress", ANSWERS_DIR)
print_table("JUDGING progress", JUDGMENTS_DIR)
