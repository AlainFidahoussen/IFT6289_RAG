# analysis

Post-hoc per-query breakdowns over all 5 subsets (1 510 queries). Pure pandas over cached judgments and rankings, no GPU required.

## Run

```bash
cd analysis && uv sync
uv run python -m analysis.per_query_type
uv run python -m analysis.easy_hard
uv run python -m analysis.paired_bootstrap
uv run python -m analysis.flips
uv run python -m analysis.stream_overlap
uv run python -m analysis.parser_output_stats
uv run python -m analysis.retrieval_value_by_type
```

Each script writes one or two CSVs to `results/` and prints the headline finding to stdout.

## Results

**pass@1 by query type (hybrid_nemo vs hybrid_deepseek, all subsets pooled)**

| Query type | n | hybrid_nemo | hybrid_deepseek | Delta |
|---|---|---|---|---|
| numerical | 297 | 73.7 | 74.1 | +0.3 |
| extractive | 540 | 81.3 | 82.6 | +1.3 |
| multi-hop | 118 | 89.8 | 91.5 | +1.7 |
| enumerative | 194 | 80.4 | 80.9 | +0.5 |
| boolean | 147 | 85.7 | 87.8 | +2.0 |
| compare-contrast | 292 | 84.6 | 85.3 | +0.7 |
| open-ended | 506 | 89.9 | 91.5 | +1.6 |

DeepSeek hybrid wins on every type, but by at most 2 pts. The expected "faithful OCR helps on numerical/tables" pattern does not appear; numerical is the smallest delta (+0.3 pts).

**Easy vs hard queries, avg pass@1 across all 5 subsets**

Easy = answered correctly by the closed-book baseline. Hard = not.

| Condition | Easy | Hard |
|---|---|---|
| jina_nemo_reranked | 91.9 | 74.7 |
| jina_deepseek_reranked | 91.0 | 75.0 |
| hybrid_deepseek | 90.6 | 73.5 |
| hybrid_nemo | 89.0 | 72.3 |

On easy queries, NeMo reranked leads and hybrid_nemo is worst; adding visual pages to NeMo hurts. On hard queries all four conditions cluster within 3 pts with no clear winner.
