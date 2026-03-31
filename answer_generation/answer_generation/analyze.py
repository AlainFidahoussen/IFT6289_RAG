"""Aggregate NDCG@10 and pass@1 into a unified comparison table."""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

app = typer.Typer()

REPO_ROOT = Path(__file__).resolve().parents[2]

# Retrieval result CSVs from sibling subprojects
RETRIEVAL_CSVS: dict[str, Path] = {
    "jina_nemo": REPO_ROOT / "textual_retriever" / "results_jina.csv",
    "jina_nemo_reranked": REPO_ROOT / "textual_retriever" / "results_jina_reranked.csv",
    "jina_deepseek": REPO_ROOT / "textual_retriever" / "results_jina_deepseek.csv",
    "jina_deepseek_reranked": REPO_ROOT / "textual_retriever" / "results_jina_reranked_deepseek.csv",
    "colembed": REPO_ROOT / "visual_retriever" / "results_colembed.csv",
}

ANSWER_RESULTS_CSV = Path("results_answers.csv")


def _load_ndcg_results() -> pd.DataFrame:
    """Load all NDCG@10 results and tag with condition name."""
    frames: list[pd.DataFrame] = []
    for condition, csv_path in RETRIEVAL_CSVS.items():
        if not csv_path.exists():
            logger.warning("Retrieval result file not found", condition=condition, path=str(csv_path))
            continue
        df = pd.read_csv(csv_path)
        df["condition"] = condition
        frames.append(df[["condition", "subset", "lang", "ndcg_at_10"]])

    if not frames:
        raise FileNotFoundError("No retrieval result CSVs found.")
    return pd.concat(frames, ignore_index=True)


def _load_answer_results() -> pd.DataFrame | None:
    """Load pass@1 results if available."""
    if not ANSWER_RESULTS_CSV.exists():
        logger.warning("Answer results file not found", path=str(ANSWER_RESULTS_CSV))
        return None
    df = pd.read_csv(ANSWER_RESULTS_CSV)
    # Keep the latest run per (condition, subset, lang)
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["condition", "subset", "lang"], keep="last")
        [["condition", "subset", "lang", "pass_at_1", "num_correct", "num_total"]]
    )


def _pivot_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to conditions × subsets with an 'avg' column."""
    pivot = df.pivot_table(index="condition", columns="subset", values=value_col, aggfunc="mean")
    pivot["avg"] = pivot.mean(axis=1)
    return pivot.round(2)


@app.command()
def main(
    output_csv: str = typer.Option("results_comparison.csv", help="Output CSV path"),
):
    """Print and save a unified NDCG@10 + pass@1 comparison table."""
    ndcg_df = _load_ndcg_results()
    answer_df = _load_answer_results()

    logger.info("NDCG@10 by condition and subset:")
    ndcg_pivot = _pivot_table(ndcg_df, "ndcg_at_10")
    print("\n=== NDCG@10 ===")
    print(ndcg_pivot.to_string())

    if answer_df is not None:
        logger.info("pass@1 by condition and subset:")
        pass1_pivot = _pivot_table(answer_df, "pass_at_1")
        print("\n=== pass@1 ===")
        print(pass1_pivot.to_string())

        # Merge for joint table
        merged = ndcg_df.merge(answer_df, on=["condition", "subset", "lang"], how="outer")
        out_path = Path(output_csv)
        merged.to_csv(out_path, index=False)
        logger.success("Joint results saved", path=str(out_path))

        # Delta: NeMo reranked vs DeepSeek reranked (the key comparison)
        conditions_of_interest = ["jina_nemo_reranked", "jina_deepseek_reranked"]
        delta_df = answer_df[answer_df["condition"].isin(conditions_of_interest)]
        if len(delta_df) >= 2:
            print("\n=== NeMo vs DeepSeek (reranked) — pass@1 delta ===")
            delta_pivot = _pivot_table(delta_df, "pass_at_1")
            print(delta_pivot.to_string())

            nemo_avg = ndcg_df[ndcg_df["condition"] == "jina_nemo_reranked"]["ndcg_at_10"].mean()
            ds_avg = ndcg_df[ndcg_df["condition"] == "jina_deepseek_reranked"]["ndcg_at_10"].mean()
            print(f"\nNDCG@10 delta (NeMo−DeepSeek): {nemo_avg - ds_avg:+.2f} pts")
    else:
        ndcg_df.to_csv(Path(output_csv), index=False)
        logger.info("NDCG@10 results saved (no answer results yet)", path=output_csv)


if __name__ == "__main__":
    app()
