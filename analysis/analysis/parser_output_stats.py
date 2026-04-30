"""Experiment 5 — parser output characterization.

Per-page counts of text length, image placeholders, and table-structure markers
for NeMo and DeepSeek-OCR-2, across the 5 subsets. Produces a per-page CSV and
per-subset summary stats.

Emits:
    analysis/results/parser_stats.csv
    analysis/results/parser_stats_summary.csv
"""

from __future__ import annotations

import re

import pandas as pd
from tqdm import tqdm

from analysis.io import (
    SUBSET_LANG,
    load_deepseek_markdown,
    load_nemo_markdown_map,
    results_path,
)

FIGURE_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
DEEPSEEK_TABLE_OPEN_RE = re.compile(r"<table>", re.IGNORECASE)
NEMO_MD_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
EQUATION_RE = re.compile(r"\\\[|\\\(|\$\$|\$[^$]+\$")
MD_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


def stats_for_text(text: str) -> dict:
    if not text:
        return {
            "chars": 0,
            "lines": 0,
            "figure_placeholders": 0,
            "html_tables": 0,
            "md_table_rows": 0,
            "equation_markers": 0,
            "md_headings": 0,
        }
    return {
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "figure_placeholders": len(FIGURE_PLACEHOLDER_RE.findall(text)),
        "html_tables": len(DEEPSEEK_TABLE_OPEN_RE.findall(text)),
        "md_table_rows": len(NEMO_MD_TABLE_ROW_RE.findall(text)),
        "equation_markers": len(EQUATION_RE.findall(text)),
        "md_headings": len(MD_HEADING_RE.findall(text)),
    }


def main() -> None:
    subsets = list(SUBSET_LANG)
    rows: list[dict] = []
    for subset in subsets:
        nemo_map = load_nemo_markdown_map(subset)
        for corpus_id, nemo_text in tqdm(nemo_map.items(), desc=f"{subset}"):
            ne = stats_for_text(nemo_text)
            ds = stats_for_text(load_deepseek_markdown(subset, corpus_id))
            rows.append(
                {
                    "subset": subset,
                    "corpus_id": int(corpus_id),
                    **{f"nemo_{k}": v for k, v in ne.items()},
                    **{f"deepseek_{k}": v for k, v in ds.items()},
                }
            )
    df = pd.DataFrame(rows)
    out = results_path("parser_stats.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} pages)")

    # Summary: per-subset medians + ratios
    stat_cols = [
        "chars", "lines", "figure_placeholders", "html_tables",
        "md_table_rows", "equation_markers", "md_headings",
    ]
    rows_sum: list[dict] = []
    for subset in subsets:
        sub = df[df["subset"] == subset]
        row = {"subset": subset, "n_pages": len(sub)}
        for c in stat_cols:
            row[f"nemo_{c}_median"] = int(sub[f"nemo_{c}"].median())
            row[f"deepseek_{c}_median"] = int(sub[f"deepseek_{c}"].median())
        row["chars_ratio_ds_over_ne"] = round(
            sub["deepseek_chars"].median() / max(sub["nemo_chars"].median(), 1), 2
        )
        rows_sum.append(row)
    summary = pd.DataFrame(rows_sum)
    out2 = results_path("parser_stats_summary.csv")
    summary.to_csv(out2, index=False)
    print(f"\nWrote {out2}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
