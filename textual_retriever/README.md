# textual-retriever

Text-stream retrieval subproject for the ift6289_rag pipeline. Uses **Jina v4** to embed ViDoRe v3 corpus markdown texts and queries, then evaluates retrieval quality with NDCG@10.

The markdown can come either from the dataset's built-in field or from the `textual-extraction` subproject (DeepSeek-OCR-2 output) — swapping the source is the core experimental manipulation.

## Role in the pipeline

```
ViDoRe v3 markdown (built-in or from textual-extraction)
        │
        ▼
 textual_retriever/dataset.py   ← precompute & cache embeddings
        │  Jina v4
        │  saves {corpus_id}.pt  (dense float32 vector)
        │  saves {query_id}.pt
        ▼
 textual_retriever/predict.py
        │  cosine similarity (L2-normalized matmul, no model needed)
        ▼
        NDCG@10
```

## Install

```bash
cd textual_retriever
uv sync
```

## Run all subsets

```bash
bash run_all.sh
```

This runs `dataset.py` + `predict.py` for all 8 public ViDoRe v3 subsets with the correct language per subset, and writes results to `results_jina.csv`. Already-cached embeddings are skipped so the script is safe to resume after interruptions.

## Run a single subset

```bash
uv run textual_retriever/dataset.py --subset computer_science --lang english
uv run textual_retriever/predict.py --subset computer_science --lang english
```

## Subsets and languages

ViDoRe v3 has 8 public subsets (2 are private hold-out sets). Each subset must be queried in its correct language:

| Subset | Language |
|---|---|
| computer_science | english |
| finance_en | english |
| pharmaceuticals | english |
| hr | english |
| industrial | english |
| physics | french |
| energy | french |
| finance_fr | french |

## Comparison with the paper

Our scores are systematically **1–4 points higher** than Table 1 of the ViDoRe v3 paper. This is expected: Table 1 evaluates cross-lingually (queries in all 6 languages against English/French documents), while this code evaluates monolingually (English queries on English subsets, French queries on French subsets). The paper itself reports a 2–3 point gap between monolingual and cross-lingual settings (Tables 9/10 vs Table 1).

## Cache layout

Embeddings are stored under `data/processed/` as individual `.pt` files:

```
jina_cache_markdowns_{subset}_{lang}/   ← corpus embeddings (language-agnostic)
jina_cache_queries_{subset}_{lang}/     ← query embeddings (language-specific)
```
