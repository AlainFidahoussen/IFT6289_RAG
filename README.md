# ift6289_rag

**Text Extraction Quality in Vision-Text Hybrid Retrieval: Controlled End-to-End QA Evaluation on Visually Rich Documents**

## Research Goal

This project is a controlled study of how **document parsing quality affects end-to-end QA** over visually rich scientific PDFs (figures, tables, equations) in a hybrid retrieval pipeline.

The variable under study is the OCR/parsing step: we swap different extractors (NeMo Retriever extraction vs. DeepSeek-OCR-2) and measure their impact on final answer accuracy, keeping the retrieval and generation components fixed.

## Pipeline

The full evaluation pipeline follows the [ViDoRe v3](https://huggingface.co/datasets/vidore/vidore_v3_physics) protocol:

```
ViDoRe v3 corpus (PDF page images + markdown)
         │
         ├─── [1] Visual retrieval  ──── ColEmbed (Nemotron) ──────────────────────┐
         │                                                                          │
         └─── [2] Parsing / OCR  ──── DeepSeek-OCR-2 ──── [3] Textual retrieval   │
                    ↑ variable                                  Jina v4             │
                                                                                    │
                                    [4] Answer generation ── Gemini-3-pro ←────────┘
                                          │
                                    [5] LLM judge ── GPT-5.2
                                    (Correct / Partially Correct / Incorrect)
```

Steps 4–5 (answer generation and LLM judging) are not yet implemented. The current code covers steps 1–3 and reports **NDCG@10** as the retrieval metric.

**Dataset:** `vidore/vidore_v3_{subset}` — 3 English subsets (`computer_science`, `finance_en`, `pharmaceuticals`), public split.

## Subprojects

The repo contains **three independent uv subprojects**, each with its own `pyproject.toml` and lockfile. All commands must be run from inside the subproject directory.

| Subproject | Path | Role |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4 using pre-existing or OCR-produced markdown |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 — produces markdown consumed by textual-retriever |

## How to Run

### 1. Visual retriever (ColEmbed)

```bash
cd visual_retriever
uv sync
uv run visual_retriever/dataset.py   # Pre-compute and cache image + query embeddings
uv run visual_retriever/predict.py   # Evaluate NDCG@10 from cached embeddings
```

### 2. Textual extraction (DeepSeek-OCR-2)

Run this before the textual retriever when evaluating OCR-produced markdown.

```bash
cd textual_extraction
uv sync
uv run textual_extraction/dataset.py  # Run OCR on corpus images, save markdown per document
```

### 3. Textual retriever (Jina v4)

```bash
cd textual_retriever
uv sync
uv run textual_retriever/dataset.py   # Pre-compute and cache markdown + query embeddings
uv run textual_retriever/predict.py   # Evaluate NDCG@10 from cached embeddings
```

### Linting / formatting (repo root)

```bash
make lint      # ruff format --check && ruff check
make format    # ruff check --fix && ruff format
make test      # pytest tests/
```

## Architecture

Each subproject follows the same layered pattern:

```
config.py    → VIDORE_SUBSET, VIDORE_LANG, cache directory names, PROCESSED_DATA_DIR
dataset.py   → load_data_vidore(): loads corpus/queries/qrels/metadata from HuggingFace
model.py     → load_*(): loads the model from HuggingFace
features.py  → precompute_*() / load_precomputed_*(): disk-cache embeddings/markdown as files
predict.py   → typer CLI: loads cached data, calls evaluate_ndcg(), logs NDCG@10
utils.py     → evaluate_ndcg(), ndcg_at_k() — pure evaluation logic
```

Embeddings are cached as individual `{id}.pt` files under `<subproject>/data/processed/<cache_dir>/`. Re-running a script after an interruption resumes from where it left off — already-cached files are skipped.

## Active Subsets

All three subprojects run the same three English subsets: `computer_science`, `finance_en`, `pharmaceuticals`.

## Project Organization

```
├── Makefile
├── README.md
├── visual_retriever/           ← ColEmbed image-stream retrieval
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── data/processed/         ← cached embeddings (gitignored)
│   └── visual_retriever/
│       ├── config.py
│       ├── dataset.py
│       ├── features.py
│       ├── model.py
│       ├── predict.py
│       └── utils.py
├── textual_retriever/          ← Jina v4 text-stream retrieval
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── data/processed/         ← cached embeddings (gitignored)
│   └── textual_retriever/
│       ├── config.py
│       ├── dataset.py
│       ├── features.py
│       ├── model.py
│       ├── predict.py
│       └── utils.py
└── textual_extraction/         ← DeepSeek-OCR-2 parsing
    ├── pyproject.toml
    ├── uv.lock
    ├── data/processed/         ← cached markdown files (gitignored)
    └── textual_extraction/
        ├── config.py
        ├── dataset.py
        ├── features.py
        └── model.py
```
