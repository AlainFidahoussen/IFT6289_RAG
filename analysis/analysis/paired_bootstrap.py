"""Experiment 2 — paired bootstrap significance for condition comparisons.

Answers: is the +1.5 pt hybrid_deepseek advantage statistically distinguishable
from zero, or within noise?

Emits analysis/results/paired_bootstrap.csv with columns:
    comparison   — e.g. "hybrid_deepseek - hybrid_nemo"
    scope        — "all" or one of the 5 subsets
    n            — number of paired queries
    delta_pts    — mean pass@1 difference (pct points)
    ci_low       — 2.5th percentile of bootstrap deltas
    ci_high      — 97.5th percentile of bootstrap deltas
    p_two_sided  — bootstrap two-sided p-value for delta != 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.io import SUBSET_LANG, load_judgments, results_path

# Comparisons to compute: (condition_a, condition_b) means A - B
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("hybrid_deepseek", "hybrid_nemo"),
    ("jina_nemo_reranked", "hybrid_deepseek"),
    ("hybrid_deepseek", "jina_deepseek_reranked"),
    ("jina_nemo_reranked", "hybrid_nemo"),
)

N_BOOT: int = 10_000
RNG = np.random.default_rng(42)


def _paired_correct_arrays(
    df: pd.DataFrame, cond_a: str, cond_b: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) binary arrays of correctness over the intersection of query_ids."""
    wide = (
        df[df["condition"].isin([cond_a, cond_b])]
        .pivot_table(index=["subset", "query_id"], columns="condition", values="correct")
        .dropna()
    )
    a = wide[cond_a].to_numpy(dtype=np.int8)
    b = wide[cond_b].to_numpy(dtype=np.int8)
    return a, b


def _bootstrap_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Paired bootstrap CI + two-sided p-value on the mean difference (a - b)."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    n = len(diff)
    obs = diff.mean()
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = diff[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    # Two-sided p by symmetry around 0 (paired-diff null distribution via reflection)
    centered = boot - obs
    p = float((np.abs(centered) >= np.abs(obs)).mean())
    return float(obs), float(ci_low), float(ci_high), p


def main() -> None:
    subsets = list(SUBSET_LANG)
    conditions_used = sorted({c for pair in COMPARISONS for c in pair})
    judgments = load_judgments(subsets, conditions_used)

    rows: list[dict] = []
    for cond_a, cond_b in COMPARISONS:
        # global (all subsets pooled)
        a, b = _paired_correct_arrays(judgments, cond_a, cond_b)
        delta, lo, hi, p = _bootstrap_ci(a, b)
        rows.append(
            {
                "comparison": f"{cond_a} - {cond_b}",
                "scope": "all",
                "n": len(a),
                "delta_pts": round(delta * 100, 3),
                "ci_low": round(lo * 100, 3),
                "ci_high": round(hi * 100, 3),
                "p_two_sided": round(p, 4),
            }
        )
        # per subset
        for subset in subsets:
            sub = judgments[judgments["subset"] == subset]
            a, b = _paired_correct_arrays(sub, cond_a, cond_b)
            if len(a) == 0:
                continue
            delta, lo, hi, p = _bootstrap_ci(a, b)
            rows.append(
                {
                    "comparison": f"{cond_a} - {cond_b}",
                    "scope": subset,
                    "n": len(a),
                    "delta_pts": round(delta * 100, 3),
                    "ci_low": round(lo * 100, 3),
                    "ci_high": round(hi * 100, 3),
                    "p_two_sided": round(p, 4),
                }
            )

    df = pd.DataFrame(rows)
    out = results_path("paired_bootstrap.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

    print("\nGlobal (all 2132 queries pooled):")
    print(df[df["scope"] == "all"].to_string(index=False))
    print("\nPer subset (only the hybrid_deepseek - hybrid_nemo comparison):")
    focus = df[(df["comparison"] == "hybrid_deepseek - hybrid_nemo") & (df["scope"] != "all")]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
