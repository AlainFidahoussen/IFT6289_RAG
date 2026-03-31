# ift6289_rag

**Text Extraction Quality in Vision-Text Hybrid Retrieval: Controlled End-to-End QA Evaluation on Visually Rich Documents**

## Research Goal

Does replacing **NeMo Retriever extraction** with **DeepSeek-OCR-2** improve retrieval quality (NDCG@10) and answer accuracy in a hybrid retrieval pipeline over visually rich PDFs?

The OCR/parsing step is the only variable. All other components — retriever, reranker, generator, judge — are held fixed across conditions.

## Pipeline

The full evaluation pipeline follows the [ViDoRe V3](https://arxiv.org/abs/2601.08620) protocol:

```
ViDoRe v3 corpus (PDF page images + markdown)
         │
         ├─── [1] Visual retrieval  ──── ColEmbed-3B-v2 ────────────────────────┐
         │                                                                        │
         └─── [2] Parsing / OCR  ──── NeMo  ─┐                                  │
                    ↑ variable                 ├── [3] Jina v4 + zerank-2 ────── │
                                  DeepSeek ───┘                                  │
                                                                                  ▼
                                              [4] Answer generation ── qwen3.5:35b (Ollama)
                                                        │
                                              [5] LLM judge ── llama3.1:8b (Ollama)
                                              (binary: Correct / Incorrect, pass@1)
```

The paper uses Gemini 3 Pro for generation and GPT-5.2 for judging. We use local models via Ollama to keep costs at zero, and use different model families for generation and judging to avoid self-evaluation bias.

### Hybrid context

For each query, the generator receives:
- **5 page images** from ColEmbed top-5 (visual stream)
- **5 markdown texts** from Jina-v4 + zerank-2 top-5 (textual stream)

Pages are concatenated without deduplication — the same page may appear in both streams (~2–3 pages overlap on average per the paper).

## Experimental Conditions

| Condition | Modality | Reranker | TOP_K |
|---|---|---|---|
| `jina_nemo` | text | — | 5 |
| `jina_nemo_reranked` | text | zerank-2 | 5 |
| `jina_deepseek` | text | — | 5 |
| `jina_deepseek_reranked` | text | zerank-2 | 5 |
| `colembed` | image | — | 5 |
| `hybrid_nemo` | hybrid | zerank-2 (text stream) | 5+5 |
| `hybrid_deepseek` | hybrid | zerank-2 (text stream) | 5+5 |

**Dataset:** `vidore/vidore_v3_{subset}` — 3 English subsets: `computer_science` (215 queries), `finance_en` (309 queries), `pharmaceuticals` (364 queries).

## Results

### Retrieval: NDCG@10

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| NeMo, no rerank | 65.23 | 50.09 | 58.60 | 57.97 |
| NeMo + zerank-2 | 83.02 | 72.73 | 69.53 | **75.09** |
| DeepSeek, no rerank | 64.03 | 46.94 | 56.48 | 55.82 |
| DeepSeek + zerank-2 | 82.37 | 65.65 | 65.05 | 71.02 |
| ColEmbed (visual) | 78.04 | 68.60 | 67.23 | 71.29 |

Paper baselines (cross-lingual avg): Jina-v4 50.4 · Jina+zerank-2 63.6 · ColEmbed 59.8. Our higher scores are expected for monolingual English evaluation.

### Answer generation: pass@1 (computer_science only so far)

| Condition | Correct | Total | pass@1 |
|---|---|---|---|
| jina_nemo | 197 | 215 | 91.6% |
| jina_deepseek | 195 | 215 | 90.7% |
| jina_nemo_reranked | 199 | 215 | 92.6% |
| jina_deepseek_reranked | 205 | 215 | **95.3%** |
| colembed | 200 | 215 | 93.0% |
| hybrid_nemo | 200 | 215 | 93.0% |
| hybrid_deepseek | 205 | 215 | **95.3%** |

Note: CS scores are inflated by parametric knowledge (48.6% of ViDoRe V3 queries are answerable without retrieval). Finance and Pharmaceuticals results pending.

## Key Findings

**DeepSeek underperforms NeMo for retrieval:**
- Without reranking: −2.15 pts avg
- With zerank-2: −4.07 pts avg (gap widens with reranking)
- Cause: DeepSeek transcribes everything verbatim (captions, footers, axis labels), diluting the semantic signal for Jina v4. NeMo produces shorter, denser text better suited for retrieval.

**Visual context adds little on easy queries:**
- hybrid_nemo == colembed (93.0%) on CS — images add no signal over text alone
- The paper finds visual context helps on *hard* queries (+2.4–2.8 pts), not easy ones
- ColEmbed and Jina+zerank-2 retrieve ~4/5 identical pages, so hybrid sends mostly duplicate context

## Subprojects

Four independent uv subprojects — **all commands must be run from inside the subproject directory.**

| Subproject | Path | Role |
|---|---|---|
| `visual-retriever` | `visual_retriever/` | Image-stream retrieval with ColEmbed (Nemotron) |
| `textual-retriever` | `textual_retriever/` | Text-stream retrieval with Jina v4; `--source nemo` or `--source deepseek` |
| `textual-extraction` | `textual_extraction/` | OCR/parsing via DeepSeek-OCR-2 |
| `answer-generation` | `answer_generation/` | Answer generation, LLM judging, and analysis |

## How to Run

### 1. Retrieval

```bash
# Visual (ColEmbed)
cd visual_retriever && uv sync && bash run_all.sh

# Textual — NeMo
cd textual_retriever && uv sync
bash run_all.sh               # → results_jina.csv
bash run_all.sh --rerank      # → results_jina_reranked.csv

# OCR extraction (required for DeepSeek conditions)
cd textual_extraction && uv sync && bash run_all.sh

# Textual — DeepSeek
cd textual_retriever
bash run_all.sh --deepseek             # → results_jina_deepseek.csv
bash run_all.sh --deepseek --rerank    # → results_jina_reranked_deepseek.csv
```

All retrieval scripts also save pre-computed per-query rankings as JSON for fast answer generation (no re-embedding needed at generation time).

### 2. Answer generation

```bash
cd answer_generation && uv sync

# Generate answers for one condition (resume-safe)
bash run_all.sh --condition jina_nemo

# Judge answers
bash run_judge.sh --condition jina_nemo

# Check progress across all conditions
uv run check_progress.py

# Final comparison table (NDCG@10 vs pass@1)
uv run answer_generation/analyze.py
```

### 3. Lint / format

```bash
make lint && make format
```

## Architecture

Each subproject follows the same layered pattern:

```
config.py   → constants, cache paths, model names
dataset.py  → load_data_vidore(): loads corpus/queries/qrels from HuggingFace
model.py    → load_*(): loads model from HuggingFace or local
features.py → precompute_*() / load_precomputed_*(): disk-cache embeddings as .pt files
predict.py  → typer CLI: evaluation or generation entry point
utils.py    → pure evaluation/generation logic
```

Embeddings cached as `{id}.pt` under `<subproject>/data/processed/<cache_dir>/`. Rankings cached as `rankings_{condition}_{subset}_{lang}.json`. All scripts are resume-safe.

## Project Layout

```
├── README.md
├── CLAUDE.md                   ← implementation notes for Claude Code
├── VIDORE.md                   ← detailed ViDoRe V3 benchmark description
├── Makefile
├── visual_retriever/           ← ColEmbed image-stream retrieval
├── textual_retriever/          ← Jina v4 text-stream retrieval + zerank-2
├── textual_extraction/         ← DeepSeek-OCR-2 parsing
└── answer_generation/          ← generation (qwen3.5:35b) + judging (llama3.1:8b)
    ├── results_answers.csv     ← pass@1 results per condition/subset
    └── check_progress.py       ← generation and judging progress table
```
