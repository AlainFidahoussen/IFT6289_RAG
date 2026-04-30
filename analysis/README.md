# analysis/

Post-hoc per-query analyses over cached ViDoRe v3 answer-generation results. Pure
pandas over the JSONs produced by the other subprojects — no generation, no
judging, no GPU.

## How to run

```bash
cd analysis
uv sync
uv run python -m analysis.per_query_type
uv run python -m analysis.paired_bootstrap
uv run python -m analysis.flips
uv run python -m analysis.stream_overlap
uv run python -m analysis.parser_output_stats
uv run python -m analysis.retrieval_value_by_type
```

Each script:
- reads cached judgments / answers / rankings via `analysis.io`,
- writes one or two CSVs into `results/`,
- prints the top-line takeaway to stdout.

## What each script answers

### 1. `per_query_type.py` — does DeepSeek's hybrid win concentrate anywhere?

Groups `hybrid_nemo` vs `hybrid_deepseek` pass@1 by query-level facets (query_types, content_type, query_format, source_type). Emits `results/per_query_type.csv`.

**Finding (all subsets pooled, n ≥ 50)**:

```
Biggest DeepSeek advantage   | Δ (pct pts) | n
-----------------------------+-------------+----
query_format=keyword         | +4.78       | 272
content_type=Mixed           | +2.55       | 157
source_type=summary          | +2.31       | 1040
content_type=Chart           | +2.29       | 175
query_types=boolean          | +2.04       | 147

Smallest / negative
---------------------------+--------+----
content_type=Other         | -3.85  |  52
source_type=image          | -0.43  | 470
query_format=instruction   | -0.42  | 473
query_types=numerical      | +0.34  | 297
content_type=Table         | +0.67  | 594
```

The "faithful OCR helps on tables and equations" intuition is **not supported**: DeepSeek's advantage on Table (+0.67) and numerical (+0.34) is essentially zero. The biggest delta is on **keyword-format** queries, and DeepSeek actually *loses* on image-sourced queries.

### 2. `paired_bootstrap.py` — is the +1.5 pt advantage real?

10k bootstrap paired-difference CIs over the 1510 paired binary outcomes. Emits `results/paired_bootstrap.csv`.

```
Comparison                              n     Δ (pts)   95% CI          p
----------------------------------------+----+--------+----------------+-------
hybrid_deepseek − hybrid_nemo           1510   +1.46    [ 0.00, 2.85]   0.054
jina_nemo_reranked − hybrid_nemo        1510   +2.52    [ 0.60, 4.37]   0.009
jina_nemo_reranked − hybrid_deepseek    1510   +1.06    [-0.73, 2.85]   0.26
hybrid_deepseek − jina_deepseek_reranked 1510   +0.00    [-1.72, 1.72]   1.0
```

Three headline findings:
- **The hybrid_deepseek vs hybrid_nemo margin is borderline** (p = 0.054, CI touches zero). Honest framing: "consistent in direction, just at the edge of significance globally."
- **Adding ColEmbed images to `jina_deepseek_reranked` adds literally nothing.** The hybrid image stream's contribution is indistinguishable from zero (Δ = 0.00, p = 1.0). The poster's "multimodal complementary evidence" hypothesis is strongly undermined.
- **`jina_nemo_reranked` beats `hybrid_nemo` significantly** (Δ = +2.52, p = 0.009). Adding the visual stream to NeMo text actively hurts.

Per-subset paired bootstrap: none of the single-subset `hybrid_deepseek − hybrid_nemo` deltas is significant individually (all p ≥ 0.086).

### 3. `flips.py` — what flips between NeMo and DeepSeek?

Classifies every query as `both_correct` / `both_wrong` / `nemo_only` / `deepseek_only` and clusters the judge's explanations. Emits `results/flip_counts.csv`, `results/flip_clusters.csv`, `results/flip_examples.md`.

**Flip counts (all 1510 paired queries)**:

```
both_correct     1224  (81%)
both_wrong        156  (10%)
deepseek_only      76  ( 5%)  DS caught it, NE missed
nemo_only          54  ( 4%)  NE caught it, DS missed
```

The +1.46 pt margin is **22 queries out of 1510**.

**Qualitative pattern from `flip_examples.md`**: DeepSeek tends to win on queries where the generator produced a **more verbose, structured answer** that the judge rewards for "accurately captures the essential information, including details X, Y, Z." NeMo tends to win when the ground truth is **short or hedged** and DeepSeek's verbose answer is marked down for being "too absolute" or "not capturing the nuance." This is as much a *judge-and-generator-style* effect as a *parser-faithfulness* effect — style dependency, not content dependency.

### 4. `stream_overlap.py` — does retrieval-stream agreement explain the finance_fr flip?

For every query, computes `|ColEmbed top-5 ∩ Jina_X_reranked top-5|` and correlates with hybrid pass@1. Emits `results/stream_overlap.csv`, `results/stream_overlap_summary.csv`.

```
Mean overlap per subset (ColEmbed vs Jina_deepseek_reranked top-5 of 5):
  computer_science 3.11
  pharmaceuticals  2.46
  physics          2.35
  finance_en       2.33
  finance_fr       2.32
```

finance_fr does have the lowest overlap but only marginally — physics and finance_en are within 0.03 of it. The "weak visual stream on finance_fr" claim has only mild support.

**Conditional hybrid Δ by overlap size (all subsets, colembed vs jina_deepseek)**:

```
overlap=0  n=107   Δ= -1.9 pts
overlap=1  n=212   Δ= +1.9
overlap=2  n=434   Δ= +1.8
overlap=3  n=418   Δ= +1.2
overlap=4  n=290   Δ= +2.8
overlap=5  n= 49   Δ= -2.0
```

Non-monotonic. No clean "more agreement → bigger parser effect" story. The "disjoint-pages dilute the prompt" hypothesis is not cleanly supported.

### 5. `parser_output_stats.py` — how do the two parsers' outputs actually differ?

Per-page counts of chars, lines, figure placeholders, table markers, headings. Emits `results/parser_stats.csv` (10,673 rows) and `results/parser_stats_summary.csv`.

```
            chars_ratio                        lines_median
            (DS / NeMo)                        NeMo   DS
cs          0.91                                20    44
finance_en  1.03                                15    30
pharma      0.98                                 8    17
physics     0.86                                 5    23
finance_fr  1.02                                27    41
```

**DeepSeek is NOT more verbose in characters** — 0.86-1.03× ratio across subsets. It is, however, consistently more *structured*: ~2× more line breaks, `##` headings on every page. So the "DeepSeek's output dilutes the embedding with more tokens" narrative from the poster's "Why the reversal?" is **factually wrong on average**. What DeepSeek actually adds is structure, not length.

### 6. `retrieval_value_by_type.py` — which queries actually need retrieval?

closed_book → hybrid_deepseek gain per query-type bin. Emits `results/retrieval_value.csv`.

**Highest retrieval value (all subsets pooled, n ≥ 50)**:

```
facet=value            | closed-book | hybrid_ds | gain
-----------------------+-------------+-----------+-------
query_types=numerical  | 36.4        | 74.1      | +37.7 pts
query_types=extractive | 54.3        | 82.6      | +28.3
source_type=image      | 60.2        | 84.9      | +24.7
content_type=Table     | 56.6        | 80.6      | +24.1
query_types=multi-hop  | 68.6        | 91.5      | +22.9

Lowest retrieval value
-----------------------+-------------+-----------+-------
query_types=open-ended | 82.8        | 91.5      | +8.7
content_type=Image     | 84.7        | 93.3      | +8.7
content_type=Mixed     | 85.4        | 89.2      | +3.8
content_type=Other     | 96.2        | 88.5      | -7.7
```

This is the cleanest result: **retrieval value is highly concentrated on numerical / extractive / Table / multi-hop queries**, and near-zero on open-ended queries. This supports a revised story for the poster: "retrieval value scales with how document-specific the question is."

## Takeaways for the poster (data-anchored, replacing speculation)

1. **The hybrid image stream appears to contribute nothing** when paired with DeepSeek text (Δ = 0.00 vs text-only). This is the most defensible new finding.
2. **`jina_nemo_reranked` > `hybrid_nemo` significantly** (+2.52 pts, p = 0.009) — adding images to NeMo hurts.
3. **The DeepSeek vs NeMo hybrid margin is borderline** (p = 0.054). Honest framing is required on the poster.
4. **Retrieval value is query-type-dependent**: +38 pts on numerical, +8 on open-ended. The current poster's per-subset closed-book-gain story is better told per-query-type.
5. **The "DeepSeek is verbose → dilutes retrieval" story is wrong on average** — DeepSeek is not longer, just more structured. The retrieval gap needs a different mechanism (an explanation we don't yet have).
6. **DeepSeek's win may be a judge-and-generator-style effect**, not a parser-faithfulness effect. Qualitative flips show DS wins by producing *more detailed answers that the judge rewards*, not by accessing content NeMo missed.

## Files produced

```
analysis/results/
├── per_query_type.csv          (113 rows)
├── paired_bootstrap.csv        (24 rows)
├── flip_counts.csv             (35 rows)
├── flip_clusters.csv           (10 rows)
├── flip_examples.md            (9 sampled flips)
├── stream_overlap.csv          (1510 rows, per query)
├── stream_overlap_summary.csv  (12 rows)
├── parser_stats.csv            (10,673 rows, per page)
├── parser_stats_summary.csv    (5 rows)
└── retrieval_value.csv         (113 rows)
```
