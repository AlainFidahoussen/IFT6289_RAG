# Text Extraction Still Matters for End-to-End QA Evaluation

A comparison of NeMo Retriever and DeepSeek-OCR-2 as document parsers in a hybrid RAG pipeline on ViDoRe v3. IFT6289 final project, Université de Montréal / MILA.

*Alain Fidahoussen · Munyeong Kim · Aftab Gazali*

**Code:** https://github.com/AlainFidahoussen/IFT6289_RAG

---

## Research question

Does parser choice affect retrieval quality (NDCG@10) and end-to-end answer accuracy (pass@1) in a hybrid RAG pipeline over visually-rich PDFs? We fix every component except the text parser and compare NeMo Retriever (retrieval-optimised, compact output) against DeepSeek-OCR-2 (faithful to visual layout, verbose output).

## Main findings

**Parser choice affects retrieval consistently.** NeMo outperforms DeepSeek on all three subsets: +2.2 pts on average without reranking and +4.0 pts with zerank-2. Finance shows the largest gap with reranking (7.0 pts).

**Reranking amplifies the parser gap.** zerank-2 boosts NeMo by +17.1 pts but DeepSeek by only +15.3 pts. Compact NeMo text gives the reranker cleaner signal; verbose OCR output dilutes it.

**Parser affects QA accuracy less and inconsistently.** Text-only with reranking: NeMo leads by 0.6 pts on average, but DeepSeek wins on CS (95.3% vs 92.6%) while NeMo wins on Finance (83.5% vs 79.0%). Retrieval quality does not translate directly into answer accuracy.

**Visual pages compensate for parser quality.** Hybrid DeepSeek (88.7%) overtakes both NeMo reranked (88.2%) and NeMo hybrid (86.9%). Adding ColEmbed pages to DeepSeek gains +1.1 pts in QA; adding them to NeMo loses -1.3 pts. Visual retrieval closes the parser gap in end-to-end accuracy.

## Pipeline

```
PDF pages
  ├── [NeMo | DeepSeek-OCR-2] → Jina v4 → (zerank-2) → top-5 text pages ─┐
  │                                                                          ├─ qwen3.5:35b → llama3.1:8b judge
  └── ColEmbed ─────────────────────────────────────────── top-5 images ───┘
```

Only the parser changes. Embedder, reranker, generator, and judge are fixed across all conditions. We use open-weights models via Ollama with different model families for generation and judging to avoid self-evaluation bias. The paper uses Gemini 3 Pro + GPT-5.2.

## Dataset

ViDoRe v3 (Loison et al., 2026), 3 English subsets.

| Subset | Queries | Pages |
|---|---|---|
| computer_science | 215 | 2 153 |
| finance_en | 309 | 1 479 |
| pharmaceuticals | 364 | 3 047 |

## Conditions

| Name | Modality | Parser | Reranker | k |
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

**Retrieval (NDCG@10)**

| | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| NeMo | 65.2 | 50.1 | 58.6 | 57.97 |
| NeMo + zerank-2 | **83.0** | **72.7** | 69.5 | **75.07** |
| DeepSeek | 64.0 | 46.9 | 56.5 | 55.80 |
| DeepSeek + zerank-2 | 82.4 | 65.7 | 65.1 | 71.07 |
| ColEmbed (images) | 78.0 | 68.6 | **67.2** | 71.27 |

NeMo leads by 2.2 pts without reranking and 4.0 pts with zerank-2. ColEmbed (no text parsing at all) ties DeepSeek+zerank-2 at 71.3 avg.

**End-to-end accuracy (pass@1 %)**

| | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| closed_book | 91.6 | 56.0 | 68.4 | 72.0 |
| jina_nemo | 91.6 | 79.6 | 82.4 | 84.5 |
| jina_deepseek | 90.7 | 73.8 | 82.4 | 82.3 |
| colembed | 93.0 | 74.8 | 83.2 | 83.7 |
| hybrid_nemo | 93.0 | 79.9 | 87.9 | 86.9 |
| jina_nemo_reranked | 92.6 | **83.5** | 88.5 | 88.2 |
| jina_deepseek_reranked | **95.3** | 79.0 | 88.5 | 87.6 |
| hybrid_deepseek | **95.3** | 81.6 | **89.3** | **88.7** |

NeMo reranked leads at 88.2%, DeepSeek reranked at 87.6%. The hybrid DeepSeek condition (88.7%) edges ahead overall; visual pages make up for weaker text retrieval on CS and pharma.

## Setup

**Python:** ≥ 3.12 for all subprojects; `textual_extraction` requires ≥ 3.13.

**Package manager:** [uv](https://docs.astral.sh/uv/). Each subproject has its own `pyproject.toml`. Inside any subproject, `uv sync` creates the virtual environment and installs all pinned dependencies (same effect as `pip install -r requirements.txt`).

**Ollama models** (needed for generation and judging):

```bash
ollama pull qwen3.5:35b   # answer generator
ollama pull llama3.1:8b   # LLM judge
```

**GPU:** ≥ 24 GB VRAM for the ColEmbed and Jina v4 embedding passes. The analysis subproject runs on CPU.

## Reproducing the experiments

All commands run from inside the subproject directory. Install dependencies once per subproject with `uv sync`.

### 1. OCR extraction (DeepSeek conditions only)

```bash
cd textual_extraction
uv sync
bash run_all.sh
```

### 2. Retrieval

```bash
cd visual_retriever && uv sync && bash run_all.sh

cd textual_retriever && uv sync
bash run_all.sh                      # NeMo, no rerank
bash run_all.sh --rerank             # NeMo + zerank-2
bash run_all.sh --deepseek           # DeepSeek, no rerank
bash run_all.sh --deepseek --rerank  # DeepSeek + zerank-2
```

### 3. Answer generation and judging

```bash
cd answer_generation && uv sync

# run for each condition you want to evaluate
bash run_all.sh  --condition jina_nemo
bash run_judge.sh --condition jina_nemo

uv run answer_generation/analyze.py  # prints final comparison table
```

### 4. Closed-book baseline

```bash
cd answer_generation_no_retrieval && uv sync && bash run_all.sh
```

### Dry run on a single subset

To test the pipeline without running all 888 queries, call the individual scripts with `--subset` and `--lang`:

```bash
# retrieval on one subset
cd textual_retriever && uv sync
uv run textual_retriever/predict.py --subset computer_science --lang en

# generation + judging on one subset
cd answer_generation && uv sync
uv run answer_generation/predict.py --subset computer_science --lang en --condition jina_nemo
uv run answer_generation/judge.py   --subset computer_science --lang en --condition jina_nemo
```

All scripts are resume-safe: already-computed results are skipped on re-run.

## Layout

```
├── visual_retriever/               ColEmbed image-stream retrieval
├── textual_retriever/              Jina v4 text-stream retrieval + zerank-2
├── textual_extraction/             DeepSeek-OCR-2 OCR pipeline
├── answer_generation/              generation + judging + analysis
├── answer_generation_no_retrieval/ closed-book baseline
└── poster/                         A0 poster (PNG)
```
