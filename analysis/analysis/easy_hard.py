"""Experiment 7 — easy / hard stratification using our own closed-book baseline.

The ViDoRe v3 paper defines a query as *easy* if any of 6 frontier LLMs answers it
correctly without retrieved context, and *hard* otherwise. They report that hybrid
retrieval wins "on hard queries" (+2.6 pts over text-only with Gemini 3 Pro).

Our generator (qwen3.5:35b) is not in that panel, and our overall pool (1510 paired
queries) may have very different easy / hard mass. This script replicates the
stratification using our own closed-book outcome and recomputes the key deltas
conditional on easy vs hard.

Easy = qwen3.5:35b Correct closed-book
Hard = qwen3.5:35b Incorrect closed-book

Emits:
  analysis/results/easy_hard_summary.csv   — deltas per (scope, difficulty, comparison)
  analysis/results/easy_hard_bootstrap.csv — paired bootstrap CIs for each delta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.io import CLOSED_BOOK_CONDITION, SUBSET_LANG, load_judgments, results_path

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("hybrid_deepseek", "jina_deepseek_reranked"),  # Does the image stream help?
    ("hybrid_deepseek", "hybrid_nemo"),             # Does parser choice help in hybrid?
    ("jina_nemo_reranked", "hybrid_nemo"),          # Do images hurt NeMo text?
    ("jina_nemo_reranked", "hybrid_deepseek"),      # Reranked-text vs hybrid_deepseek
)

N_BOOT: int = 10_000
RNG = np.random.default_rng(42)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    n = len(diff)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0)
    obs = float(diff.mean())
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = diff[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    centered = boot - obs
    p = float((np.abs(centered) >= np.abs(obs)).mean())
    return obs, float(ci_low), float(ci_high), p


def main() -> None:
    subsets = list(SUBSET_LANG)
    conditions = sorted({c for pair in COMPARISONS for c in pair} | {CLOSED_BOOK_CONDITION})
    judgments = load_judgments(subsets, conditions)

    # Per-query closed-book label → "easy" if correct, else "hard"
    closed = judgments[judgments["condition"] == CLOSED_BOOK_CONDITION][
        ["subset", "query_id", "correct"]
    ].rename(columns={"correct": "closed_book_correct"})
    merged = judgments[judgments["condition"] != CLOSED_BOOK_CONDITION].merge(
        closed, on=["subset", "query_id"], how="left"
    )
    merged["difficulty"] = np.where(merged["closed_book_correct"] == 1, "easy", "hard")

    # -------- Summary: mean pass@1 per (subset, difficulty, condition) --------
    summary = (
        merged.groupby(["subset", "difficulty", "condition"])
        .agg(n=("correct", "size"), pass_at_1=("correct", "mean"))
        .reset_index()
    )
    summary["pass_at_1"] = (summary["pass_at_1"] * 100).round(2)
    out_summary = results_path("easy_hard_summary.csv")
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {out_summary}")

    # Pretty printout of difficulty mix per subset
    print("\nEasy / hard mix per subset (our closed-book):")
    mix = (
        merged[merged["condition"] == merged["condition"].iloc[0]]
        .groupby(["subset", "difficulty"])
        .size()
        .unstack(fill_value=0)
    )
    mix["total"] = mix.sum(axis=1)
    mix["pct_hard"] = (mix.get("hard", 0) / mix["total"] * 100).round(1)
    print(mix.to_string())

    # -------- Paired bootstrap per (scope, difficulty, comparison) --------
    rows: list[dict] = []
    for cond_a, cond_b in COMPARISONS:
        for scope_label, scope_df in [("all", merged)] + [
            (s, merged[merged["subset"] == s]) for s in subsets
        ]:
            for diff in ("easy", "hard", "both"):
                if diff == "both":
                    sub = scope_df
                else:
                    sub = scope_df[scope_df["difficulty"] == diff]
                # Pivot to paired arrays
                wide = (
                    sub[sub["condition"].isin([cond_a, cond_b])]
                    .pivot_table(
                        index=["subset", "query_id"],
                        columns="condition",
                        values="correct",
                    )
                    .dropna()
                )
                if len(wide) == 0:
                    continue
                a = wide[cond_a].to_numpy(dtype=np.int8)
                b = wide[cond_b].to_numpy(dtype=np.int8)
                obs, lo, hi, p = _bootstrap_ci(a, b)
                rows.append(
                    {
                        "scope": scope_label,
                        "difficulty": diff,
                        "comparison": f"{cond_a} - {cond_b}",
                        "n": len(a),
                        "delta_pts": round(obs * 100, 2),
                        "ci_low": round(lo * 100, 2),
                        "ci_high": round(hi * 100, 2),
                        "p_two_sided": round(p, 4),
                    }
                )

    boot_df = pd.DataFrame(rows)
    out_boot = results_path("easy_hard_bootstrap.csv")
    boot_df.to_csv(out_boot, index=False)
    print(f"\nWrote {out_boot}")

    # -------- Top-line takeaways --------
    print("\n=== GLOBAL (all subsets pooled) by difficulty ===")
    print(
        boot_df[boot_df["scope"] == "all"]
        .sort_values(["comparison", "difficulty"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
