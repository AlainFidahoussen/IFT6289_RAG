# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

**Does replacing NeMo Retriever extraction with DeepSeek-OCR-2 improve retrieval quality (NDCG@10) and answer accuracy in a hybrid retrieval pipeline over visually rich PDFs?**

Full pipeline (ViDoRe v3 protocol):
1. **Retrieval**: ColEmbed (image stream) + Jina v4 with zerank-2 reranker (text stream)
2. **Parsing/OCR** ← *the variable*: NeMo Retriever extraction vs. DeepSeek-OCR-2
3. **Answer generation**: local Qwen3.5:35b via Ollama (paper uses Gemini 3 Pro)
4. **Evaluation**: local Qwen3.5:35b LLM judge, binary label (Correct / Incorrect), pass@1

Dataset: ViDoRe v3 (`vidore/vidore_v3_{subset}`) — 3 English subsets: `computer_science`, `finance_en`, `pharmaceuticals`.

## Experimental Conditions

| Condition | Modality | TOP_K | Reranker |
|---|---|---|---|
| `jina_nemo` | text | 5 | — |
| `jina_nemo_reranked` | text | 5 | zerank-2 |
| `jina_deepseek` | text | 5 | — |
| `jina_deepseek_reranked` | text | 5 | zerank-2 |
| `colembed` | image | 5 | — |
| `hybrid_nemo` | hybrid | 5+5=10 | zerank-2 (text stream) |
| `hybrid_deepseek` | hybrid | 5+5=10 | zerank-2 (text stream) |

**Paper protocol (Section 4.2):** top-5 pages for all single-stream conditions; hybrid concatenates top-5 visual (images) + top-5 textual (text), no deduplication.

## Model Choices

| Role | Model | Rationale |
|---|---|---|
| Generator | `qwen3.5:35b` | Strong multimodal reasoning, runs locally via Ollama |
| Judge | `llama3.1:8b` | Different family from generator — avoids self-evaluation bias |

**Why different families matter:** using the same model (or same family) for both generation and judging introduces self-preference bias — the judge favors answers that match its own style and shares the same blind spots. A smaller model from a different family is a better judge than a larger model from the same family. The paper uses Gemini 3 Pro (generator) + GPT-5.2 (judge) for the same reason. Binary correct/incorrect judgment with the ground-truth answer provided is a simple enough task for an 8B model.

## NDCG@10 Results

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| NeMo, no rerank | 65.23 | 50.09 | 58.60 | 57.97 |
| NeMo + zerank-2 | 83.01 | 72.76 | 69.50 | 75.09 |
| DeepSeek, no rerank | 64.03 | 46.94 | 56.48 | 55.82 |
| DeepSeek + zerank-2 | 82.37 | 65.65 | 65.05 | 71.02 |
| ColEmbed (visual) | 78.04 | 68.60 | 67.23 | 71.29 |

Paper baselines (cross-lingual avg, all subsets): Jina-v4 50.4 · Jina+zerank-2 63.6 · ColEmbed 59.8. Our higher scores are expected for monolingual English evaluation.

## Why DeepSeek Underperforms NeMo

Without reranking: −2.15 pts avg. With zerank-2: −4.07 pts avg (gap widens).

1. **OCR output is noisy and long** — dilutes semantic signal for Jina v4. NeMo produces shorter, denser text.
2. **NeMo was optimized for retrieval; DeepSeek for faithfulness** — ViDoRe v3 paper (Section 4.1, footnote 3) notes NeMo's simplest pipeline performed best.
3. **Reranking amplifies the gap** — zerank-2 lifts NeMo +17.1 pts but DeepSeek only +15.2 pts.

## Subprojects

Four independent uv subprojects — **all commands must be run from inside the subproject directory.**

| Subproject | Path | Role |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4 |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 |
| `answer-generation` | `answer_generation/` | Generation + judging + analysis |

## Commands

```bash
# Retrieval (produces cached rankings JSON + results CSV)
cd visual_retriever  && bash run_all.sh
cd textual_retriever && bash run_all.sh
cd textual_retriever && bash run_all.sh --rerank
cd textual_retriever && bash run_all.sh --deepseek
cd textual_retriever && bash run_all.sh --deepseek --rerank

# OCR extraction (only needed for DeepSeek conditions)
cd textual_extraction && bash run_all.sh

# Answer generation (per condition)
cd answer_generation && bash run_all.sh --condition jina_nemo

# LLM judging (after generation)
cd answer_generation && bash run_judge.sh --condition jina_nemo

# Final comparison table
cd answer_generation && uv run answer_generation/analyze.py

# Lint / format (repo root)
make lint && make format
```

## Architecture

Each subproject follows the same layered pattern:

```
config.py   → constants, cache directory names, PROCESSED_DATA_DIR
dataset.py  → load_data_vidore(): loads corpus/queries/qrels from HuggingFace
model.py    → load_*(): loads the model from HuggingFace
features.py → precompute_*() / load_precomputed_*(): disk-cache embeddings as .pt files
predict.py  → typer CLI: loads cached data, runs evaluation or generation
utils.py    → pure evaluation/generation logic
```

Embeddings cached as `{id}.pt` under `<subproject>/data/processed/<cache_dir>/`. Rankings cached as `rankings_{condition}_{subset}_{lang}.json`. All scripts are resume-safe.

- **Visual retriever**: ColPali-style multi-vector late interaction. Corpus embeddings `[T, d]` scored with MaxSim.
- **Textual retriever**: cosine similarity on L2-normalized dense vectors (pure numpy, no model at eval time). Optional zerank-2 reranks top-100 candidates.
- **Answer generation**: `answer_generation/features.py` loads cached rankings (fast path), then `predict.py` calls Ollama. `judge.py` scores answers. `analyze.py` produces final comparison.
- **Corpus IDs**: ViDoRe v3 corpus_ids are contiguous 0 to N-1 — rankings store positions which are valid corpus_ids.

Evaluation metric: NDCG@10 with graded relevance (0/1/2), reported on a 0–100 scale.
