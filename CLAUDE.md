# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

**Controlled study of how document parsing quality affects end-to-end QA** over visually rich scientific PDFs in hybrid retrieval pipelines.

Following the ViDoRe v3 evaluation protocol, the full pipeline is:
1. **Retrieval**: ColEmbed (image stream) + Jina v4 with zerank-2 reranker (text stream)
2. **Parsing/OCR** ← *the variable being studied*: swap NeMo Retriever extraction vs. DeepSeek-OCR 2 (and others)
3. **Answer generation**: Gemini-3-pro conditioned on retrieved page image + extracted text
4. **Evaluation**: GPT-5.2 judge (Correct / Partially Correct / Incorrect) using the exact ViDoRe v3 rubric

Dataset: ViDoRe v3 (`vidore/vidore_v3_{subset}`) — 5 English + 3 French subsets, public split.

## Subprojects

The repo contains **three independent uv subprojects**, each with its own `pyproject.toml` and lockfile. **All commands must be run from inside the subproject directory.**

| Subproject | Path | Role in pipeline |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4 using pre-existing markdown |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 to produce markdown from document images |

Zerank-2 reranking, answer generation (Gemini-3-pro), and LLM judging (GPT-5.2) are not yet implemented in code.

## Commands

### Visual Retriever (ColEmbed / Nemotron)
```bash
cd visual_retriever
uv sync
uv run visual_retriever/dataset.py   # Pre-compute and cache image + query embeddings
uv run visual_retriever/predict.py   # Evaluate NDCG@10 from cached embeddings
```

### Textual Retriever (Jina v4)
```bash
cd textual_retriever
uv sync
uv run textual_retriever/dataset.py  # Pre-compute and cache markdown + query embeddings
uv run textual_retriever/predict.py  # Evaluate NDCG@10 from cached embeddings
```

### Textual Extraction (DeepSeek-OCR-2)
```bash
cd textual_extraction
uv sync
uv run textual_extraction/dataset.py  # Run OCR on corpus images, save markdown per document
```

### Linting / Formatting (from repo root)
```bash
make lint      # ruff format --check && ruff check
make format    # ruff check --fix && ruff format
make test      # pytest tests/
```

## Architecture

Each subproject follows the same layered pattern:

```
config.py   → VIDORE_SUBSET, VIDORE_LANG, cache directory names, PROCESSED_DATA_DIR
dataset.py  → load_data_vidore(): loads corpus/queries/qrels/metadata from HuggingFace
model.py    → load_*(): loads the model from HuggingFace
features.py → precompute_*() / load_precomputed_*(): disk-cache embeddings as .pt files
predict.py  → typer CLI: loads cached embeddings, calls evaluate_ndcg(), logs NDCG@10
utils.py    → evaluate_ndcg(), ndcg_at_k() — pure evaluation logic
```

### Data Flow

1. **Embeddings/markdown are precomputed and cached** to `<subproject>/data/processed/<cache_dir>/` as individual `{id}.pt` files (one per corpus document or query). This avoids re-running expensive model inference.
2. **`predict.py` loads from cache only** — it does not re-run the embedding model.
3. **`textual_extraction`** is a preprocessing step: it runs DeepSeek-OCR-2 over corpus images to produce markdown, which is then embedded and consumed by `textual_retriever`.

### Active Subsets (configured in `config.py`)
- `visual_retriever/config.py`: `VIDORE_SUBSET = "physics"`
- `textual_retriever/config.py` and `textual_extraction/config.py`: `VIDORE_SUBSET = "computer_science"`

### Key Differences Between Retrievers

- **Visual retriever**: uses `model.get_scores(query_emb, pages_emb)` — ColPali-style multi-vector late interaction via `colpali-engine`. Corpus embeddings are multi-vector tensors `[T, d]`.
- **Textual retriever**: uses cosine similarity on L2-normalized dense vectors. No model is needed at eval time — pure numpy matmul in `utils.evaluate_ndcg`.
- **Textual extraction**: uses `model.infer(tokenizer, prompt, image_file, ...)` from DeepSeek-OCR-2. Saves markdown text (not embeddings) to disk; embeddings are computed separately by `textual_retriever`.

### Evaluation Metric

NDCG@10 with graded relevance (0/1/2) per ViDoRe convention. Reported on a 0–100 scale. The end-to-end QA metric (once implemented) is answer accuracy from the GPT-5.2 LLM judge.
