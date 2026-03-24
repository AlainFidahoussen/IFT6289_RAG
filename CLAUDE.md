# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

**Does replacing NeMo Retriever extraction with DeepSeek-OCR-2 improve retrieval quality (NDCG@10) and answer accuracy in a hybrid retrieval pipeline over visually rich PDFs?**

Full pipeline (ViDoRe v3 protocol):
1. **Retrieval**: ColEmbed (image stream) + Jina v4 with zerank-2 reranker (text stream)
2. **Parsing/OCR** ← *the variable*: NeMo Retriever extraction vs. DeepSeek-OCR-2
3. **Answer generation**: multimodal VLM (Gemini-3-pro in the paper; local Qwen2.5-VL-72B-4bit)
4. **Evaluation**: GPT-5.2 judge (medium reasoning effort), binary label (Correct / Incorrect), pass@1

Dataset: ViDoRe v3 (`vidore/vidore_v3_{subset}`) — 3 English subsets: `computer_science`, `finance_en`, `pharmaceuticals`. English-only to avoid cross-lingual confounds; three domains to cover different visual content types (equations/code, tables, mixed figures).

## Experimental Conditions

| # | Condition | Reranker | Result file |
|---|---|---|---|
| 1 | Jina v4, NeMo | — | `textual_retriever/results_jina.csv` |
| 2 | Jina v4, NeMo | zerank-2 | `textual_retriever/results_jina_reranked.csv` |
| 3 | Jina v4, DeepSeek | — | `textual_retriever/results_jina_deepseek.csv` |
| 4 | Jina v4, DeepSeek | zerank-2 | `textual_retriever/results_jina_reranked_deepseek.csv` *(pending)* |
| 5 | ColEmbed (visual) | — | `visual_retriever/results_colembed.csv` |

## NDCG@10 Results

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| NeMo, no rerank | 65.23 | 50.09 | 58.60 | 57.97 |
| NeMo + zerank-2 | 83.01 | 72.76 | 69.50 | 75.09 |
| DeepSeek, no rerank | 63.67 | 46.62 | 56.74 | 55.68 |
| DeepSeek + zerank-2 | — | — | — | *(pending)* |
| ColEmbed (visual) | 78.04 | 68.60 | 67.23 | 71.29 |

Paper baselines (cross-lingual avg, all subsets): Jina-v4 50.4 · Jina+zerank-2 63.6 · ColEmbed 59.8. Our higher scores are expected for monolingual English evaluation.

## Why DeepSeek Underperforms NeMo (−2.3 pts avg)

1. **OCR output is noisy and long.** DeepSeek transcribes everything visible (captions, axis labels, footers, page numbers), which dilutes the semantic signal seen by Jina v4. NeMo produces shorter, denser text.
2. **NeMo was optimized for retrieval; DeepSeek for faithfulness.** The ViDoRe v3 paper (Section 4.1, footnote 3) notes that even NeMo's simplest pipeline (no image descriptions) performed best — retrieval benefits from concise text.

**Next actions (in order):**
1. Delete DeepSeek embedding cache and re-embed with clean markdown
2. Run `--deepseek --rerank` (zerank-2; expected to close much of the gap)
3. Implement answer generation + LLM judging

See `status.txt` for current task state.

## Subprojects

Three independent uv subprojects — **all commands must be run from inside the subproject directory.**

| Subproject | Path | Role |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4; `--source nemo` or `--source deepseek` |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 |

## Commands

```bash
# Visual retrieval
cd visual_retriever && bash run_all.sh                         # → results_colembed.csv

# Textual retrieval — NeMo
cd textual_retriever && bash run_all.sh                        # → results_jina.csv
cd textual_retriever && bash run_all.sh --rerank               # → results_jina_reranked.csv

# OCR extraction
cd textual_extraction && bash run_all.sh                       # → data/processed/<subset>/

# Textual retrieval — DeepSeek
cd textual_retriever && bash run_all.sh --deepseek             # → results_jina_deepseek.csv
cd textual_retriever && bash run_all.sh --deepseek --rerank    # → results_jina_reranked_deepseek.csv

# Lint / format (repo root)
make lint && make format
```

## Architecture

Each subproject follows the same layered pattern:

```
config.py   → VIDORE_SUBSET, VIDORE_LANG, cache directory names, PROCESSED_DATA_DIR
dataset.py  → load_data_vidore(): loads corpus/queries/qrels from HuggingFace
model.py    → load_*(): loads the model from HuggingFace
features.py → precompute_*() / load_precomputed_*(): disk-cache embeddings as .pt files
predict.py  → typer CLI: loads cached data, calls evaluate_ndcg(), logs NDCG@10
utils.py    → evaluate_ndcg(), ndcg_at_k() — pure evaluation logic
```

Embeddings are cached as `{id}.pt` files under `<subproject>/data/processed/<cache_dir>/`. Scripts are resume-safe — already-cached files are skipped.

- **Visual retriever**: ColPali-style multi-vector late interaction (`colpali-engine`). Corpus embeddings are `[T, d]` tensors scored with `model.get_scores()`.
- **Textual retriever**: cosine similarity on L2-normalized dense vectors. Eval is pure numpy matmul — no model at eval time. Optional zerank-2 reranks top-100 candidates.
- **Textual extraction**: `model.infer(tokenizer, prompt, image_file, ...)`. Saves markdown to disk; embeddings computed separately by `textual_retriever`.

Evaluation metric: NDCG@10 with graded relevance (0/1/2), reported on a 0–100 scale.
