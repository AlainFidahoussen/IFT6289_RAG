# visual-retriever

Image-stream retrieval subproject for the ift6289_rag pipeline. Uses **ColEmbed (Nemotron)** — a ColPali-style multi-vector model — to embed ViDoRe v3 corpus page images and queries, then evaluates retrieval quality with NDCG@10.

## Role in the pipeline

```
ViDoRe v3 corpus images + queries
        │
        ▼
 visual_retriever/dataset.py   ← precompute & cache embeddings
        │  ColEmbed (Nemotron)
        │  saves {corpus_id}.pt  (multi-vector [T, d] tensor, bfloat16)
        │  saves {query_id}.pt
        ▼
 visual_retriever/predict.py
        │  ColPali late-interaction scores
        ▼
        NDCG@10
```

## Install

```bash
cd visual_retriever
uv sync
```

## Run all subsets

```bash
bash run_all.sh
```

This runs `dataset.py` + `predict.py` for the three active English subsets, writes results to `results_colembed.csv`, and saves per-query rankings to `data/processed/rankings_colembed_{subset}_{lang}.json`. Already-cached embeddings are skipped so the script is safe to resume after interruptions.

## Run a single subset

```bash
uv run visual_retriever/dataset.py --subset computer_science --lang english
uv run visual_retriever/predict.py --subset computer_science --lang english --save-rankings
```

## Active subsets

Three English subsets are used for this study:

| Subset | Language |
|---|---|
| computer_science | english |
| finance_en | english |
| pharmaceuticals | english |

## Results (NDCG@10)

| Subset | NDCG@10 |
|---|---|
| computer_science | 78.04 |
| finance_en | 68.60 |
| pharmaceuticals | 67.23 |
| **avg** | **71.29** |

Paper baseline (cross-lingual avg): 59.8. Our higher scores are expected for monolingual English evaluation (paper reports 2–3 pt gap between monolingual and cross-lingual settings).

## Cache layout

Embeddings are stored under `data/processed/` as individual `.pt` files:

```
colembed_cache_pages_{subset}_{lang}/    ← corpus image embeddings (language-agnostic)
colembed_cache_queries_{subset}_{lang}/  ← query embeddings (language-specific)
```

Note: corpus image embeddings are language-agnostic (same pages regardless of query language), but the cache directory name includes the language suffix because it is tied to a specific `dataset.py` run.
