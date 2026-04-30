"""Experiment 1 — per-query-type pass@1 breakdown.

Answers: where does DeepSeek actually help end-to-end?

Emits analysis/results/per_query_type.csv with columns:
    facet              — "query_types" or "content_type" or "query_format" or "source_type"
    facet_value        — e.g. "numerical", "Table", "question", "image"
    subset             — "all" or one of the 5 subsets
    n                  — number of queries in the bin
    pass_at_1_nemo     — hybrid_nemo pass@1 (%)
    pass_at_1_deepseek — hybrid_deepseek pass@1 (%)
    delta              — deepseek − nemo, pct points

Also prints the top-line takeaway.
"""

from __future__ import annotations

import pandas as pd

from analysis.io import (
    RETRIEVAL_CONDITIONS,
    SUBSET_LANG,
    load_all_query_metadata,
    load_judgments,
    results_path,
)

FACETS_SCALAR: tuple[str, ...] = ("query_format", "source_type")
FACETS_LIST: tuple[str, ...] = ("query_types", "content_type")


def _explode_list_column(metadata: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explode a list-valued metadata column so each list element becomes its own row."""
    return metadata[["query_id", "subset", col]].explode(col).rename(
        columns={col: "facet_value"}
    )


def compute_breakdown(
    judgments: pd.DataFrame,
    metadata: pd.DataFrame,
    facet: str,
    scalar: bool,
) -> pd.DataFrame:
    """Return per-facet pass@1 for hybrid_nemo and hybrid_deepseek."""
    if scalar:
        meta_long = metadata[["query_id", "subset", facet]].rename(columns={facet: "facet_value"})
    else:
        meta_long = _explode_list_column(metadata, facet)

    j = judgments[judgments["condition"].isin(["hybrid_nemo", "hybrid_deepseek"])]
    merged = j.merge(meta_long, on=["query_id", "subset"], how="inner")

    rows: list[dict] = []
    for scope, sub in [("all", merged)] + [(s, merged[merged["subset"] == s]) for s in sorted(SUBSET_LANG)]:
        pivot = sub.pivot_table(
            index="facet_value",
            columns="condition",
            values="correct",
            aggfunc=["sum", "count"],
        )
        if pivot.empty:
            continue
        for facet_value, counts in pivot.iterrows():
            n_ne = int(counts[("count", "hybrid_nemo")])
            n_ds = int(counts[("count", "hybrid_deepseek")])
            if n_ne == 0 or n_ds == 0:
                continue
            pass_ne = 100.0 * counts[("sum", "hybrid_nemo")] / n_ne
            pass_ds = 100.0 * counts[("sum", "hybrid_deepseek")] / n_ds
            rows.append(
                {
                    "facet": facet,
                    "facet_value": facet_value,
                    "subset": scope,
                    "n": n_ne,  # NE and DS sample sizes are identical per query
                    "pass_at_1_nemo": round(pass_ne, 1),
                    "pass_at_1_deepseek": round(pass_ds, 1),
                    "delta": round(pass_ds - pass_ne, 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    subsets = list(SUBSET_LANG)
    judgments = load_judgments(subsets, list(RETRIEVAL_CONDITIONS))
    metadata = load_all_query_metadata(subsets)

    frames: list[pd.DataFrame] = []
    for f in FACETS_SCALAR:
        frames.append(compute_breakdown(judgments, metadata, f, scalar=True))
    for f in FACETS_LIST:
        frames.append(compute_breakdown(judgments, metadata, f, scalar=False))

    df = pd.concat(frames, ignore_index=True)
    out = results_path("per_query_type.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

    # Takeaway: top 5 and bottom 5 delta bins on the "all" scope, n >= 50
    summary = df[(df["subset"] == "all") & (df["n"] >= 50)].sort_values("delta", ascending=False)
    print("\nTop DeepSeek-favored bins (n≥50, all subsets pooled):")
    print(summary.head(8).to_string(index=False))
    print("\nBottom DeepSeek-favored bins (n≥50, all subsets pooled):")
    print(summary.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
