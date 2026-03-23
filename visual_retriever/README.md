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

This runs `dataset.py` + `predict.py` for the three active English subsets and writes results to `results_colembed.csv`. Already-cached embeddings are skipped so the script is safe to resume after interruptions.

## Run a single subset

```bash
uv run visual_retriever/dataset.py --subset computer_science --lang english
uv run visual_retriever/predict.py --subset computer_science --lang english
```

## Active subsets

Three English subsets are used for this study:

| Subset | Language |
|---|---|
| computer_science | english |
| finance_en | english |
| pharmaceuticals | english |

## Comparison with the paper

Our scores are systematically **1–4 points higher** than Table 1 of the ViDoRe v3 paper. This is expected: Table 1 evaluates cross-lingually (queries in all 6 languages against English/French documents), while this code evaluates monolingually (English queries on English subsets, French queries on French subsets). The paper itself reports a 2–3 point gap between monolingual and cross-lingual settings (Tables 9/10 vs Table 1).

## Cache layout

Embeddings are stored under `data/processed/` as individual `.pt` files:

```
colembed_cache_pages_{subset}_{lang}/    ← corpus image embeddings (language-agnostic)
colembed_cache_queries_{subset}_{lang}/  ← query embeddings (language-specific)
```

Note: corpus image embeddings are language-agnostic (same pages regardless of query language), but the cache directory name includes the language suffix because it is tied to a specific `dataset.py` run.
