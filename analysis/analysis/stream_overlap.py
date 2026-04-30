"""Experiment 4 — ColEmbed ↔ Jina reranked top-5 overlap vs hybrid pass@1.

For each hybrid query, measure how many of the top-5 pages agree between the
visual stream (ColEmbed) and the reranked text stream (Jina+zerank-2). Correlate
with hybrid pass@1 outcomes.

If the finance_fr counterexample is driven by "disjoint pages dilute the prompt",
then low-overlap queries should show smaller (or reversed) hybrid_deepseek gain.

Emits:
    analysis/results/stream_overlap.csv — per-query overlap + both hybrid correctness
    analysis/results/stream_overlap_summary.csv — aggregated by subset and overlap_size
"""

from __future__ import annotations

import pandas as pd

from analysis.io import SUBSET_LANG, load_judgments, load_rankings, results_path

TOP_K: int = 5


def compute_overlap(subsets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for subset in subsets:
        lang = SUBSET_LANG[subset]
        visual = load_rankings("colembed", subset, lang)
        jn_ne = load_rankings("jina_nemo_reranked", subset, lang)
        jn_ds = load_rankings("jina_deepseek_reranked", subset, lang)
        for qid, vis_top in visual.items():
            vis5 = set(vis_top[:TOP_K])
            txt5_ne = set(jn_ne[qid][:TOP_K]) if qid in jn_ne else set()
            txt5_ds = set(jn_ds[qid][:TOP_K]) if qid in jn_ds else set()
            rows.append(
                {
                    "subset": subset,
                    "query_id": int(qid),
                    "overlap_colembed_nemo": len(vis5 & txt5_ne),
                    "overlap_colembed_deepseek": len(vis5 & txt5_ds),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    subsets = list(SUBSET_LANG)
    overlaps = compute_overlap(subsets)

    # Join with hybrid_nemo & hybrid_deepseek correctness
    j = load_judgments(subsets, ["hybrid_nemo", "hybrid_deepseek"])
    pivot = j.pivot_table(
        index=["subset", "query_id"], columns="condition", values="correct"
    ).reset_index()
    pivot.columns = [
        c if isinstance(c, str) else c for c in pivot.columns  # flatten single-index
    ]
    merged = overlaps.merge(pivot, on=["subset", "query_id"], how="inner")
    merged["delta"] = merged["hybrid_deepseek"] - merged["hybrid_nemo"]

    out = results_path("stream_overlap.csv")
    merged.to_csv(out, index=False)
    print(f"Wrote {out} ({len(merged)} rows)")

    # -------- Aggregates --------
    print("\nMean overlap per subset (ColEmbed vs Jina_nemo_reranked top-5):")
    print(merged.groupby("subset")["overlap_colembed_nemo"].mean().round(2).to_string())

    print("\nMean overlap per subset (ColEmbed vs Jina_deepseek_reranked top-5):")
    print(merged.groupby("subset")["overlap_colembed_deepseek"].mean().round(2).to_string())

    print("\nConditional pass@1 by ColEmbed↔Jina_nemo overlap size (all subsets):")
    summary_ne = (
        merged.groupby("overlap_colembed_nemo")
        .agg(
            n=("query_id", "size"),
            hybrid_nemo=("hybrid_nemo", "mean"),
            hybrid_deepseek=("hybrid_deepseek", "mean"),
            delta=("delta", "mean"),
        )
        .round(3)
    )
    summary_ne["hybrid_nemo"] = (summary_ne["hybrid_nemo"] * 100).round(1)
    summary_ne["hybrid_deepseek"] = (summary_ne["hybrid_deepseek"] * 100).round(1)
    summary_ne["delta"] = (summary_ne["delta"] * 100).round(1)
    print(summary_ne.to_string())

    print("\nConditional pass@1 by ColEmbed↔Jina_deepseek overlap size (all subsets):")
    summary_ds = (
        merged.groupby("overlap_colembed_deepseek")
        .agg(
            n=("query_id", "size"),
            hybrid_nemo=("hybrid_nemo", "mean"),
            hybrid_deepseek=("hybrid_deepseek", "mean"),
            delta=("delta", "mean"),
        )
        .round(3)
    )
    summary_ds["hybrid_nemo"] = (summary_ds["hybrid_nemo"] * 100).round(1)
    summary_ds["hybrid_deepseek"] = (summary_ds["hybrid_deepseek"] * 100).round(1)
    summary_ds["delta"] = (summary_ds["delta"] * 100).round(1)
    print(summary_ds.to_string())

    # Persist the aggregate
    summary_ne["overlap_stream"] = "colembed_vs_nemo"
    summary_ds["overlap_stream"] = "colembed_vs_deepseek"
    summary = pd.concat(
        [summary_ne.reset_index(), summary_ds.reset_index()], ignore_index=True
    )
    out2 = results_path("stream_overlap_summary.csv")
    summary.to_csv(out2, index=False)
    print(f"\nWrote {out2}")


if __name__ == "__main__":
    main()
