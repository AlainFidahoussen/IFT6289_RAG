"""Experiment 3 — flip analysis between hybrid_nemo and hybrid_deepseek.

For every query, compare the two judgments. Bucket into:
    both_correct, both_wrong, NE→DS_wrong (NeMo right, DS wrong), DS→NE_wrong (DS right, NeMo wrong).

Emits:
    analysis/results/flip_counts.csv  — counts per (subset, query_types_primary, direction)
    analysis/results/flip_examples.md — 3 sampled queries per direction with gt, both answers, judge reasoning
    analysis/results/flip_clusters.csv — TF-IDF + KMeans (k=5) clusters of judge explanations per direction

The goal is to characterize WHAT kinds of failure each parser produces.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis.io import (
    SUBSET_LANG,
    load_all_query_metadata,
    load_judgments,
    results_path,
)

N_CLUSTERS: int = 5
N_EXAMPLES_PER_DIRECTION: int = 3
SAMPLE_SEED: int = 0


def build_paired_frame(judgments: pd.DataFrame) -> pd.DataFrame:
    """Wide-pivot hybrid_nemo + hybrid_deepseek onto one row per query."""
    rows = judgments[judgments["condition"].isin(["hybrid_nemo", "hybrid_deepseek"])]
    # Pivot on multiple columns via a manual merge for clarity
    ne = rows[rows["condition"] == "hybrid_nemo"].set_index(["subset", "query_id"])
    ds = rows[rows["condition"] == "hybrid_deepseek"].set_index(["subset", "query_id"])
    merged = ne[["query", "gt_answer", "predicted_answer", "judgment", "explanation", "correct"]].join(
        ds[["predicted_answer", "judgment", "explanation", "correct"]],
        lsuffix="_nemo",
        rsuffix="_deepseek",
        how="inner",
    ).reset_index()
    # Flip direction:
    def _dir(row: pd.Series) -> str:
        ne_c = bool(row["correct_nemo"])
        ds_c = bool(row["correct_deepseek"])
        if ne_c and ds_c:
            return "both_correct"
        if (not ne_c) and (not ds_c):
            return "both_wrong"
        if ne_c and (not ds_c):
            return "nemo_only"  # NeMo caught it, DeepSeek missed
        return "deepseek_only"  # DeepSeek caught it, NeMo missed

    merged["flip"] = merged.apply(_dir, axis=1)
    return merged


def cluster_explanations(texts: list[str], n_clusters: int = N_CLUSTERS) -> tuple[list[int], list[str]]:
    """TF-IDF + KMeans on the judge's explanations; return labels + top keywords per cluster."""
    if len(texts) < n_clusters:
        return [0] * len(texts), ["<too few>"] * n_clusters
    vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    terms = vec.get_feature_names_out()
    cluster_terms: list[str] = []
    for ci in range(n_clusters):
        order = km.cluster_centers_[ci].argsort()[::-1][:6]
        cluster_terms.append(", ".join(terms[order].tolist()))
    return km.labels_.tolist(), cluster_terms


def main() -> None:
    subsets = list(SUBSET_LANG)
    judgments = load_judgments(subsets, ["hybrid_nemo", "hybrid_deepseek"])
    metadata = load_all_query_metadata(subsets)[
        ["query_id", "subset", "query_types_primary", "content_type_primary"]
    ]

    paired = build_paired_frame(judgments)
    paired = paired.merge(metadata, on=["query_id", "subset"], how="left")

    # ---- flip_counts.csv ----
    counts = (
        paired.groupby(["subset", "query_types_primary", "flip"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    out_counts = results_path("flip_counts.csv")
    counts.to_csv(out_counts, index=False)
    print(f"Wrote {out_counts}")
    print("\nGlobal flip counts (across all 2132 queries):")
    print(paired["flip"].value_counts().to_string())
    print("\nPer subset:")
    print(paired.groupby(["subset", "flip"]).size().unstack(fill_value=0).to_string())

    # ---- flip_clusters.csv ----
    cluster_rows: list[dict] = []
    for direction in ("nemo_only", "deepseek_only"):
        sub = paired[paired["flip"] == direction]
        if len(sub) == 0:
            continue
        # Cluster the winning condition's explanation (i.e. the condition that got it right
        # — its explanation says WHAT it captured that the loser missed).
        expl_col = "explanation_nemo" if direction == "nemo_only" else "explanation_deepseek"
        texts = sub[expl_col].fillna("").astype(str).tolist()
        labels, cluster_terms = cluster_explanations(texts)
        for cid in range(N_CLUSTERS):
            count = labels.count(cid)
            cluster_rows.append(
                {
                    "direction": direction,
                    "cluster_id": cid,
                    "size": count,
                    "top_terms": cluster_terms[cid] if cid < len(cluster_terms) else "",
                }
            )
    cluster_df = pd.DataFrame(cluster_rows)
    out_clust = results_path("flip_clusters.csv")
    cluster_df.to_csv(out_clust, index=False)
    print(f"\nWrote {out_clust}")
    print(cluster_df.to_string(index=False))

    # ---- flip_examples.md ----
    lines: list[str] = ["# Flip examples (hybrid_nemo vs hybrid_deepseek)\n"]
    for direction in ("deepseek_only", "nemo_only"):
        sub = paired[paired["flip"] == direction]
        sample = sub.sample(
            n=min(N_EXAMPLES_PER_DIRECTION, len(sub)),
            random_state=SAMPLE_SEED,
        )
        header = {
            "deepseek_only": "## DeepSeek correct, NeMo wrong",
            "nemo_only": "## NeMo correct, DeepSeek wrong",
        }[direction]
        lines.append(header + f"  (n={len(sub)} total)\n")
        for _, row in sample.iterrows():
            lines.append(
                f"### {row['subset']} / query_id={row['query_id']} "
                f"({row['query_types_primary']}, content={row['content_type_primary']})"
            )
            lines.append(f"**Q**: {row['query']}")
            lines.append(f"**GT**: {row['gt_answer']}")
            lines.append(f"**NeMo answer**: {row['predicted_answer_nemo']}")
            lines.append(f"**DeepSeek answer**: {row['predicted_answer_deepseek']}")
            lines.append(f"**Judge on NeMo**: {row['explanation_nemo']}")
            lines.append(f"**Judge on DeepSeek**: {row['explanation_deepseek']}\n")
    out_md = results_path("flip_examples.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
