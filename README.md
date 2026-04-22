# Does adding image retrieval help a strong text-based RAG?

**Reproducing ViDoRe v3 with open-weights models** — IFT6289 Final Project, Université de Montréal / MILA

*Alain Fidahoussen · Munyeong Kim · Aftab Gazali*

---

## Research question

With the text stream fixed at **DeepSeek-OCR-2 + Jina v4 + zerank-2**, does adding a visual retrieval stream (**ColEmbed**) improve end-to-end answer accuracy?

## Pipeline

```
PDF page images
    │
    ├── DeepSeek-OCR-2 → Jina v4 → zerank-2 → Top-5 text pages ──┐
    │                                                               ├── qwen3.5:35b → llama3.1:8b judge
    └── ColEmbed ────────────────────────────── Top-5 image pages ─┘
```

We evaluate 7 conditions across 5 ViDoRe v3 subsets (3 EN + 2 FR), measuring retrieval quality (NDCG@10) and end-to-end answer accuracy (pass@1).

The paper uses Gemini 3 Pro + GPT-5.2. We use open-weights models via Ollama to keep costs at zero, with different model families for generation and judging to avoid self-evaluation bias.

## Dataset

ViDoRe v3 (Loison et al., 2026): 10 corpora, 26 000 pages, 3 099 human-verified queries in 6 languages.

| Subset | Language | Queries | Pages |
|---|---|---|---|
| computer_science | EN | 215 | 2 153 |
| finance_en | EN | 309 | 1 479 |
| pharmaceuticals | EN | 364 | 3 047 |
| physics | FR | 302 | 2 025 |
| finance_fr | FR | 320 | 1 969 |
| **total** | | **1 510** | **10 673** |

## Experimental conditions

| Condition | Modality | Reranker | TOP_K |
|---|---|---|---|
| `closed_book` | none | — | — |
| `colembed` | image | — | 5 |
| `jina_nemo` | text | — | 5 |
| `jina_nemo_reranked` | text | zerank-2 | 5 |
| `jina_deepseek` | text | — | 5 |
| `jina_deepseek_reranked` | text | zerank-2 | 5 |
| `hybrid_nemo` | hybrid | zerank-2 (text) | 5+5 |
| `hybrid_deepseek` | hybrid | zerank-2 (text) | 5+5 |

## Results

### Retrieval quality (NDCG@10)

| Subset | ColEmbed (image) | Jina + zerank-2 (text) |
|---|---|---|
| computer_science (EN) | 78.0 | **82.4** |
| finance_en (EN) | **68.6** | 65.7 |
| pharmaceuticals (EN) | **67.2** | 65.1 |
| physics (FR) | **47.8** | 46.6 |
| finance_fr (FR) | 47.6 | **48.1** |
| **avg** | **61.8** | 61.6 |

### End-to-end accuracy (pass@1 %)

| Subset | Closed-book | ColEmbed | Jina + zerank-2 | Hybrid |
|---|---|---|---|---|
| computer_science | 91.6 | 93.0 | 95.3 | **95.3** |
| finance_en | 56.0 | 74.8 | 79.0 | **81.6** |
| pharmaceuticals | 68.4 | 83.2 | 88.5 | **89.3** |
| physics | 87.8 | 83.1 | **90.1** | 89.7 |
| finance_fr | 40.6 | 70.3 | **80.3** | 77.2 |
| **avg** | 68.9 | 80.9 | 86.6 | **86.6** |

### pass@1 by query type (all subsets pooled)

| Query type | Closed | Image | Text | Hybrid |
|---|---|---|---|---|
| numerical | 36.4 | 70.4 | 70.7 | **74.1** |
| extractive | 54.3 | 77.4 | **83.1** | 82.6 |
| multi-hop | 68.6 | 84.7 | 84.7 | **91.5** |
| enumerative | 60.3 | 75.8 | **83.5** | 80.9 |
| boolean | 72.1 | 85.0 | **93.2** | 87.8 |
| compare-contrast | 71.9 | 80.8 | 84.6 | **85.3** |
| open-ended | 82.8 | 82.6 | **91.9** | 91.5 |

## Key findings

- **Hybrid and reranked text are tied at 86.6% overall**, but they disagree on ~180 queries — the query mix determines the better choice.
- **Image retrieval helps on multi-hop (+6.8 pts) and numerical (+3.4 pts)**, where evidence spans multiple pages or requires visual table layout.
- **Image retrieval hurts on boolean (−5.4 pts)**: extra image pages dilute the prompt when the text stream already retrieves the fact.
- **Retrieval is most valuable on document-specific queries**: +37.7 pts on numerical, +28.3 on extractive, only +8.7 on open-ended over closed-book.
- **The open-source generator is likely the bottleneck**: ViDoRe v3 reports +2.6 pts hybrid gain with Gemini 3 Pro — our null result likely reflects a weaker vision stack, not uninformative images.

## Project layout

```
├── visual_retriever/           ColEmbed image-stream retrieval
├── textual_retriever/          Jina v4 text-stream retrieval + zerank-2
├── textual_extraction/         DeepSeek-OCR-2 parsing
├── answer_generation/          Answer generation + LLM judging + analysis
├── answer_generation_no_retrieval/   Closed-book baseline
├── analysis/                   Post-hoc per-query analyses (pandas, no GPU)
└── poster/                     A0 poster (HTML + PDF + PPTX)
```

All code is organised as independent `uv` subprojects. Commands must be run from inside the subproject directory.

## How to run

### Retrieval

```bash
cd visual_retriever  && uv sync && bash run_all.sh

cd textual_retriever && uv sync
bash run_all.sh                        # NeMo, no rerank
bash run_all.sh --rerank               # NeMo + zerank-2
bash run_all.sh --deepseek             # DeepSeek, no rerank
bash run_all.sh --deepseek --rerank    # DeepSeek + zerank-2
```

### OCR extraction (required for DeepSeek conditions)

```bash
cd textual_extraction && uv sync && bash run_all.sh
```

### Answer generation and judging

```bash
cd answer_generation && uv sync

bash run_all.sh --condition hybrid_deepseek   # generate answers
bash run_judge.sh --condition hybrid_deepseek  # judge answers

uv run answer_generation/analyze.py           # final comparison table
```

### Closed-book baseline

```bash
cd answer_generation_no_retrieval && uv sync && bash run_all.sh
```

### Post-hoc analysis

```bash
cd analysis && uv sync
uv run python -m analysis.per_query_type
uv run python -m analysis.paired_bootstrap
uv run python -m analysis.easy_hard
uv run python -m analysis.retrieval_value_by_type
```

### Poster

```bash
# Regenerate PDF from HTML
cd /tmp && node poster_to_pdf.mjs

# Regenerate PPTX
python3 poster/generate_pptx.py
```
