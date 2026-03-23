# textual-retriever

Text-stream retrieval subproject for the ift6289_rag pipeline. Uses **Jina v4** to embed ViDoRe v3 corpus markdown texts and queries, then evaluates retrieval quality with NDCG@10.

The markdown can come either from the dataset's built-in field (NeMo Retriever extraction, as used in the ViDoRe v3 paper) or from the `textual-extraction` subproject (DeepSeek-OCR-2 output) — swapping the source is the core experimental manipulation.

## Role in the pipeline

```
ViDoRe v3 markdown (NeMo built-in or DeepSeek-OCR-2 from textual-extraction)
        │
        ▼
 textual_retriever/dataset.py   ← precompute & cache embeddings
        │  Jina v4
        │  saves {corpus_id}.pt  (dense float32 vector)
        │  saves {query_id}.pt
        ▼
 textual_retriever/predict.py
        │  cosine similarity (L2-normalized matmul, no model needed)
        │  optional: zerank-2 reranker
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
bash run_all.sh [--deepseek] [--rerank]
```

| Command | Markdown source | Reranker | Results file |
|---|---|---|---|
| `bash run_all.sh` | NeMo (built-in) | — | `results_jina.csv` |
| `bash run_all.sh --rerank` | NeMo (built-in) | zerank-2 | `results_jina_reranked.csv` |
| `bash run_all.sh --deepseek` | DeepSeek-OCR-2 | — | `results_jina_deepseek.csv` |
| `bash run_all.sh --deepseek --rerank` | DeepSeek-OCR-2 | zerank-2 | `results_jina_reranked_deepseek.csv` |

Already-cached embeddings are skipped so the script is safe to resume after interruptions. The `--rerank` flag skips the `dataset.py` embedding step (embeddings must already be cached).

## Run a single subset

```bash
uv run textual_retriever/dataset.py --subset computer_science --lang english [--source deepseek]
uv run textual_retriever/predict.py --subset computer_science --lang english [--source deepseek] [--rerank]
```

## Active subsets

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
jina_cache_markdowns_{subset}_{lang}/           ← NeMo corpus embeddings
jina_cache_markdowns_deepseek_{subset}_{lang}/  ← DeepSeek corpus embeddings
jina_cache_queries_{subset}_{lang}/             ← query embeddings (shared across sources)
```

Query embeddings are shared between NeMo and DeepSeek runs — only the corpus embeddings differ.
