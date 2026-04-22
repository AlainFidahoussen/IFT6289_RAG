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

We evaluate 7 conditions (+ closed-book baseline) across 5 ViDoRe v3 subsets (3 EN + 2 FR), measuring retrieval quality (NDCG@10) and end-to-end answer accuracy (pass@1).

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

| Condition | Modality | Parser | Reranker | TOP_K |
|---|---|---|---|---|
| `closed_book` | none | — | — | — |
| `jina_nemo` | text | NeMo | — | 5 |
| `jina_nemo_reranked` | text | NeMo | zerank-2 | 5 |
| `jina_deepseek` | text | DeepSeek-OCR-2 | — | 5 |
| `jina_deepseek_reranked` | text | DeepSeek-OCR-2 | zerank-2 | 5 |
| `colembed` | image | — | — | 5 |
| `hybrid_nemo` | hybrid | NeMo | zerank-2 (text) | 5+5 |
| `hybrid_deepseek` | hybrid | DeepSeek-OCR-2 | zerank-2 (text) | 5+5 |

## Results

### Retrieval quality (NDCG@10)

| Condition | CS | Finance EN | Pharma | Physics | Finance FR | avg |
|---|---|---|---|---|---|---|
| Jina + NeMo | 65.2 | 50.1 | 58.6 | 43.7 | 34.5 | 50.4 |
| Jina + NeMo + zerank-2 | **83.0** | **72.7** | 69.5 | **49.1** | 53.8 | **65.6** |
| Jina + DeepSeek | 64.0 | 46.9 | 56.5 | 40.6 | 31.2 | 47.8 |
| Jina + DeepSeek + zerank-2 | 82.4 | 65.7 | 65.1 | 46.6 | 48.1 | 61.5 |
| ColEmbed (image) | 78.0 | 68.6 | **67.2** | 47.8 | **47.6** | 61.8 |

NeMo wins retrieval: +4.1 pts avg over DeepSeek with reranking. ColEmbed and NeMo+zerank-2 are nearly tied on average (61.8 vs 65.6), with ColEmbed winning on 3 of 5 subsets without reranking.

### End-to-end accuracy (pass@1 %)

| Condition | CS | Finance EN | Pharma | Physics | Finance FR | avg |
|---|---|---|---|---|---|---|
| closed_book | 91.6 | 56.0 | 68.4 | 87.8 | 40.6 | 68.9 |
| jina_nemo | 91.6 | 79.6 | 82.4 | 87.1 | 73.8 | 82.9 |
| jina_deepseek | 90.7 | 73.8 | 82.4 | 83.8 | 73.4 | 80.8 |
| colembed | 93.0 | 74.8 | 83.2 | 83.1 | 70.3 | 80.9 |
| hybrid_nemo | 93.0 | 79.9 | 87.9 | 86.8 | 77.8 | 85.1 |
| jina_nemo_reranked | 92.6 | 83.5 | **88.5** | **92.1** | 80.9 | **87.5** |
| jina_deepseek_reranked | **95.3** | 79.0 | **88.5** | 90.1 | 80.3 | 86.6 |
| hybrid_deepseek | **95.3** | **81.6** | **89.3** | 89.7 | 77.2 | 86.6 |

Despite NeMo winning retrieval, **DeepSeek hybrid matches NeMo reranked** on average (86.6% vs 87.5%) and wins on CS and finance_en. The best single condition overall is `jina_nemo_reranked` at 87.5%.

### pass@1 by query type — all subsets pooled

| Query type | Closed | Image | Text (NeMo+R) | Hybrid (DS) |
|---|---|---|---|---|
| numerical | 36.4 | 70.4 | 70.7 | **74.1** |
| extractive | 54.3 | 77.4 | **83.1** | 82.6 |
| multi-hop | 68.6 | 84.7 | 84.7 | **91.5** |
| enumerative | 60.3 | 75.8 | **83.5** | 80.9 |
| boolean | 72.1 | 85.0 | **93.2** | 87.8 |
| compare-contrast | 71.9 | 80.8 | 84.6 | **85.3** |
| open-ended | 82.8 | 82.6 | **91.9** | 91.5 |

## Key findings

- **NeMo wins retrieval** (+4.1 pts NDCG@10 avg over DeepSeek with reranking). Shorter, denser NeMo text is better suited to dense embedding.
- **Hybrid and reranked text are both at ~86–87% overall**, but they disagree on ~180 queries — the query mix determines the better choice.
- **Image retrieval helps on multi-hop (+6.8 pts) and numerical (+3.4 pts)**, where evidence spans multiple pages or requires visual table layout.
- **Image retrieval hurts on boolean (−5.4 pts)**: extra image pages dilute the prompt when the text stream already retrieves the fact.
- **Retrieval is most valuable on document-specific queries**: +37.7 pts on numerical, +28.3 on extractive, only +8.7 on open-ended over closed-book.
- **The open-source generator is likely a bottleneck**: ViDoRe v3 reports +2.6 pts hybrid gain with Gemini 3 Pro — our smaller gain likely reflects a weaker vision stack.

## Project layout

```
├── visual_retriever/               ColEmbed image-stream retrieval
├── textual_retriever/              Jina v4 text-stream retrieval + zerank-2
├── textual_extraction/             DeepSeek-OCR-2 parsing
├── answer_generation/              Answer generation + LLM judging + analysis
├── answer_generation_no_retrieval/ Closed-book baseline
├── analysis/                       Post-hoc per-query analyses (pandas, no GPU)
└── poster/                         A0 poster (HTML + PDF + PPTX)
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

bash run_all.sh --condition hybrid_deepseek    # generate answers
bash run_judge.sh --condition hybrid_deepseek   # judge answers

uv run answer_generation/analyze.py            # final comparison table
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
