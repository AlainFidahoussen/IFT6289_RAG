# ift6289_rag

**Text Extraction Quality in Vision-Text Hybrid Retrieval: Controlled End-to-End QA Evaluation on Visually Rich Documents**

## Research Goal

This project is a controlled study of how **document parsing quality affects end-to-end QA** over visually rich scientific PDFs (figures, tables, equations) in a hybrid retrieval pipeline.

The variable under study is the OCR/parsing step: we swap different extractors (**NeMo Retriever extraction** vs. **DeepSeek-OCR-2**) and measure their impact on retrieval quality (NDCG@10) and final answer accuracy, keeping all other components fixed.

## Pipeline

The full evaluation pipeline follows the [ViDoRe v3](https://huggingface.co/datasets/vidore/vidore_v3_physics) protocol:

```
ViDoRe v3 corpus (PDF page images + markdown)
         │
         ├─── [1] Visual retrieval  ──── ColEmbed (Nemotron) ──────────────────────┐
         │                                                                          │
         └─── [2] Parsing / OCR  ──── DeepSeek-OCR-2 ──── [3] Textual retrieval   │
                    ↑ variable                                  Jina v4             │
                                                           (+ zerank-2 reranker)   │
                                                                                    │
                                    [4] Answer generation ── Gemini-3-pro ←────────┘
                                          │
                                    [5] LLM judge ── GPT-5.2
                                    (Correct / Partially Correct / Incorrect)
```

Steps 4–5 (answer generation and LLM judging) are not yet implemented. The current code covers steps 1–3 and reports **NDCG@10** as the retrieval metric.

### Hybrid context fed to the generator

For each query, the generation model receives:
- **5 page images** from ColEmbed top-5 (visual stream)
- **5 markdown texts** from Jina-v4+zerank-2 top-5 (textual stream)
- The query itself

Pages are concatenated without removing duplicates — the same page may appear in both streams. On average ~7.35 unique pages per query (Table 11 of the paper), meaning ~2–3 pages appear in both streams.

### LLM judge

The paper uses **GPT-5.2 with medium reasoning effort** as the judge. It returns a **binary label** (Correct / Incorrect) in a pass@1 setting. The judge achieves Krippendorff's α = 0.91 across 5 independent runs, confirming high consistency.

### Generation model

The paper uses **Gemini 3 Pro** for the hybrid row (best non-oracle result in Table 3). Since Gemini 3 Pro is costly, the local alternative for this project is a multimodal VLM that fits in the available GPU (RTXA6000, 48GB VRAM). The generation model must accept both images and text in the same prompt.

**Local alternatives fitting in 48GB:**

| Model | VRAM (4-bit) | VRAM (bf16) | Notes |
|---|---|---|---|
| Qwen2.5-VL-72B | ~36GB | ~144GB | Best open-source quality; needs 4-bit quant |
| Qwen2.5-VL-32B | ~16GB | ~64GB | Good balance; fits in 8-bit too |
| InternVL2.5-26B | ~13GB | ~52GB | Strong alternative |
| Qwen2.5-VL-7B | ~4GB | ~14GB | Fits natively in bf16; weaker |

Qwen3-VL models may also be available (the paper uses Qwen3-VL-235B-A22B for bounding box evaluation); check HuggingFace for current sizes.

**Dataset:** `vidore/vidore_v3_{subset}` — 3 English subsets (`computer_science`, `finance_en`, `pharmaceuticals`), public split.

## Experimental Conditions

Two textual extraction conditions are compared, with visual retrieval (ColEmbed) kept fixed:

| Condition | Extractor | Markdown source |
|---|---|---|
| **Baseline** | NeMo Retriever extraction (NVIDIA Ingest) | Built into the ViDoRe v3 HuggingFace dataset |
| **Experimental** | DeepSeek-OCR-2 | Re-extracted by `textual_extraction/` subproject |

### How each extractor handles images

**NeMo Retriever extraction** uses the simplest pipeline: it extracts text from each PDF page as markdown. Images (charts, figures, diagrams) are effectively ignored — no descriptions are generated. The ViDoRe v3 paper (footnote 3, Section 4.1) explicitly notes: *"Chunking within pages or providing image descriptions did not improve our results. Thus, we report the results of the simplest pipeline."*

**DeepSeek-OCR-2** is a dedicated document OCR model that processes the page as an image and produces markdown. It faithfully transcribes visible text including tables, code, and equations. For purely visual content (charts, photographs) it produces minimal output, similar to NeMo. The expected advantage is on pages with **complex layouts** — dense tables, multi-column text, equations — where OCR fidelity matters most.

### Where to expect differences

- **Text-heavy pages with complex layout**: DeepSeek-OCR-2 may produce more accurate markdown, leading to better Jina v4 embeddings and higher NDCG@10.
- **Image-dominant pages**: Neither extractor produces meaningful text; the visual stream (ColEmbed) handles these.
- **End-to-end QA**: Even if NDCG@10 is similar, DeepSeek OCR may yield better answer accuracy by providing the LLM with more faithful page content. This is where the OCR quality signal is most likely to show up.
- **Content type breakdown**: Per Figure 6 of the ViDoRe v3 paper, Image and Mixed content types score lowest even for the best visual retriever — the textual stream cannot recover these regardless of OCR quality.

### Key findings from the ViDoRe v3 paper (baseline)

From Table 1 and Table 2 (cross-lingual evaluation, NeMo extraction):

- Jina-v4 textual retrieval: **50.4** avg NDCG@10
- Jina-v4 + zerank-2 reranker: **63.6** avg NDCG@10 (+13.2 points — largest gain of any configuration)
- ColEmbed-3B-v2 visual retrieval: **59.8** avg NDCG@10
- Visual retrievers consistently outperform textual ones at equivalent model size
- Textual reranking (zerank-2) yields far larger gains than visual reranking (+13.2 vs +0.2)

Our monolingual scores (English queries on English subsets) are expected to be **1–4 points higher** than Table 1, consistent with the paper's own monolingual vs. cross-lingual gap (Tables 9/10 vs. Table 1).

## Subprojects

The repo contains **three independent uv subprojects**, each with its own `pyproject.toml` and lockfile. All commands must be run from inside the subproject directory.

| Subproject | Path | Role |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4; supports `--source nemo` or `--source deepseek` |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 — produces markdown consumed by textual-retriever |

## How to Run

### 1. Visual retriever (ColEmbed)

```bash
cd visual_retriever
uv sync
bash run_all.sh   # embed + evaluate all three subsets → results_colembed.csv
```

### 2. Textual retriever — NeMo baseline

```bash
cd textual_retriever
uv sync
bash run_all.sh                # no rerank → results_jina.csv
bash run_all.sh --rerank       # + zerank-2 → results_jina_reranked.csv
```

### 3. Textual extraction (DeepSeek-OCR-2)

```bash
cd textual_extraction
uv sync
bash run_all.sh   # run OCR on all three subsets, save markdown to data/processed/
```

### 4. Textual retriever — DeepSeek condition

```bash
cd textual_retriever
bash run_all.sh --deepseek              # no rerank → results_jina_deepseek.csv
bash run_all.sh --deepseek --rerank     # + zerank-2 → results_jina_reranked_deepseek.csv
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
