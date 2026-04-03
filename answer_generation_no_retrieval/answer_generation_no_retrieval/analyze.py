"""Compare closed-book pass@1 against retrieval conditions from answer_generation."""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

app = typer.Typer()

REPO_ROOT = Path(__file__).resolve().parents[2]

# Closed-book results (this subproject)
CLOSED_BOOK_RESULTS_CSV = Path("results_answers.csv")

# Retrieval results (sibling subproject)
RETRIEVAL_RESULTS_CSV = REPO_ROOT / "answer_generation" / "results_answers.csv"


def _load_results(csv_path: Path) -> pd.DataFrame | None:
    """Load pass@1 results if available."""
    if not csv_path.exists():
        logger.warning("Results file not found", path=str(csv_path))
        return None
    df = pd.read_csv(csv_path)
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["condition", "subset", "lang"], keep="last")
        [["condition", "subset", "lang", "pass_at_1", "num_correct", "num_total"]]
    )


def _pivot_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to conditions x subsets with an 'avg' column."""
    pivot = df.pivot_table(index="condition", columns="subset", values=value_col, aggfunc="mean")
    pivot["avg"] = pivot.mean(axis=1)
    return pivot.round(4)


@app.command()
def main(
    output_csv: str = typer.Option(
        "results_comparison_no_retrieval.csv", help="Output CSV path"
    ),
):
    """Print closed-book vs retrieval pass@1 comparison."""
    closed_book_df = _load_results(CLOSED_BOOK_RESULTS_CSV)
    retrieval_df = _load_results(RETRIEVAL_RESULTS_CSV)

    if closed_book_df is None:
        logger.error("No closed-book results found. Run generation and judging first.")
        raise typer.Exit(code=1)

    print("\n=== Closed-book pass@1 ===")
    cb_pivot = _pivot_table(closed_book_df, "pass_at_1")
    print(cb_pivot.to_string())

    if retrieval_df is not None:
        print("\n=== Retrieval pass@1 (from answer_generation) ===")
        ret_pivot = _pivot_table(retrieval_df, "pass_at_1")
        print(ret_pivot.to_string())

        combined = pd.concat([closed_book_df, retrieval_df], ignore_index=True)
        print("\n=== All conditions pass@1 ===")
        all_pivot = _pivot_table(combined, "pass_at_1")
        print(all_pivot.to_string())

        # Delta: each retrieval condition vs closed-book
        cb_avg = cb_pivot["avg"].iloc[0]
        print(f"\n=== Retrieval lift over closed-book (avg pass@1 = {cb_avg:.4f}) ===")
        for condition in ret_pivot.index:
            delta = ret_pivot.loc[condition, "avg"] - cb_avg
            print(f"  {condition:30s}  {delta:+.4f}")

        combined.to_csv(Path(output_csv), index=False)
        logger.success("Combined results saved", path=output_csv)
    else:
        logger.warning("No retrieval results found for comparison.")
        closed_book_df.to_csv(Path(output_csv), index=False)


if __name__ == "__main__":
    app()
