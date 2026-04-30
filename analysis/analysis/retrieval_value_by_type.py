"""Experiment 6 — retrieval value (closed-book → best retrieval) by query type.

Which query types actually benefit from retrieval? If numerical queries gain +40
pts while boolean queries gain +5, the "document-specific facts" story becomes
data-anchored.

Emits analysis/results/retrieval_value.csv with columns:
    facet, facet_value, subset, n,
    closed_book_pass, hybrid_deepseek_pass, gain_pts.
"""

from __future__ import annotations

import pandas as pd

from analysis.io import (
    CLOSED_BOOK_CONDITION,
    SUBSET_LANG,
    load_all_query_metadata,
    load_judgments,
    results_path,
)

# hybrid_deepseek is the "best-retrieval" condition chosen for the comparison
BEST_RETRIEVAL_CONDITION: str = "hybrid_deepseek"
FACETS_SCALAR: tuple[str, ...] = ("query_format", "source_type")
FACETS_LIST: tuple[str, ...] = ("query_types", "content_type")


def compute(
    judgments: pd.DataFrame,
    metadata: pd.DataFrame,
    facet: str,
    scalar: bool,
) -> pd.DataFrame:
    if scalar:
        meta_long = metadata[["query_id", "subset", facet]].rename(columns={facet: "facet_value"})
    else:
        meta_long = (
            metadata[["query_id", "subset", facet]]
            .explode(facet)
            .rename(columns={facet: "facet_value"})
        )

    merged = judgments.merge(meta_long, on=["query_id", "subset"], how="inner")
    rows: list[dict] = []
    scopes = [("all", merged)] + [(s, merged[merged["subset"] == s]) for s in sorted(SUBSET_LANG)]
    for scope, sub in scopes:
        pivot = sub.pivot_table(
            index="facet_value",
            columns="condition",
            values="correct",
            aggfunc=["sum", "count"],
        )
        if pivot.empty:
            continue
        for facet_value, counts in pivot.iterrows():
            try:
                n = int(counts[("count", CLOSED_BOOK_CONDITION)])
            except KeyError:
                continue
            if n == 0:
                continue
            cb = 100.0 * counts[("sum", CLOSED_BOOK_CONDITION)] / n
            try:
                br_n = int(counts[("count", BEST_RETRIEVAL_CONDITION)])
                br = 100.0 * counts[("sum", BEST_RETRIEVAL_CONDITION)] / br_n
            except KeyError:
                continue
            rows.append(
                {
                    "facet": facet,
                    "facet_value": facet_value,
                    "subset": scope,
                    "n": n,
                    "closed_book_pass": round(cb, 1),
                    "hybrid_deepseek_pass": round(br, 1),
                    "gain_pts": round(br - cb, 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    subsets = list(SUBSET_LANG)
    judgments = load_judgments(subsets, [CLOSED_BOOK_CONDITION, BEST_RETRIEVAL_CONDITION])
    metadata = load_all_query_metadata(subsets)

    frames: list[pd.DataFrame] = []
    for f in FACETS_SCALAR:
        frames.append(compute(judgments, metadata, f, scalar=True))
    for f in FACETS_LIST:
        frames.append(compute(judgments, metadata, f, scalar=False))

    df = pd.concat(frames, ignore_index=True)
    out = results_path("retrieval_value.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

    # Takeaway: top gains per facet across all subsets (n≥50)
    print("\nWhere retrieval adds the most (all subsets pooled, n≥50):")
    top = df[(df["subset"] == "all") & (df["n"] >= 50)].sort_values("gain_pts", ascending=False)
    print(top.head(10).to_string(index=False))
    print("\nWhere retrieval helps the least:")
    print(top.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
