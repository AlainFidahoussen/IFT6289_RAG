# answer-generation

Answer generation, LLM judging, and analysis subproject for the ift6289_rag pipeline. Runs a multimodal generator (qwen3.5:35b) over retrieved pages, scores answers with a judge model (llama3.1:8b), and produces the final pass@1 comparison table.

## Role in the pipeline

```
Cached rankings from textual_retriever/ and visual_retriever/
        │
        ▼
 answer_generation/features.py   ← load top-k corpus IDs per query (fast path)
        │  TOP_K=5 per stream
        ▼
 answer_generation/predict.py    ← generate answers via Ollama
        │  qwen3.5:35b
        │  text: 5 markdown pages | image: 5 page images | hybrid: both
        ▼
 answer_generation/judge.py      ← score answers via Ollama
        │  llama3.1:8b (different family from generator — avoids self-evaluation bias)
        │  binary: Correct / Incorrect
        ▼
        pass@1  →  results_answers.csv
```

## Install

```bash
cd answer_generation
uv sync
```

Requires Ollama running locally with `qwen3.5:35b` and `llama3.1:8b` pulled:
```bash
ollama pull qwen3.5:35b
ollama pull llama3.1:8b
```

## Run

```bash
# Generate answers for one condition across all 3 subsets (resume-safe)
bash run_all.sh --condition jina_nemo

# Available conditions:
#   jina_nemo, jina_nemo_reranked,
#   jina_deepseek, jina_deepseek_reranked,
#   colembed, hybrid_nemo, hybrid_deepseek

# Judge answers
bash run_judge.sh --condition jina_nemo

# Check generation and judging progress across all conditions
uv run check_progress.py

# Final NDCG@10 vs pass@1 comparison table
uv run answer_generation/analyze.py
```

## Results (pass@1)

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| jina_nemo | 91.6% | 79.6% | 82.4% | 84.5% |
| jina_nemo_reranked | 92.6% | 83.5% | 88.5% | **88.2%** |
| jina_deepseek | 90.7% | 73.8% | 82.4% | 82.3% |
| jina_deepseek_reranked | 95.3% | 79.0% | 88.5% | 87.6% |
| colembed | 93.0% | 74.8% | 83.2% | 83.7% |
| hybrid_nemo | 93.0% | 79.9% | 87.9% | 86.9% |
| hybrid_deepseek | **95.3%** | **81.6%** | **89.3%** | **88.7%** |

## Configuration

Key constants in `answer_generation/config.py`:

| Parameter | Value | Notes |
|---|---|---|
| `GENERATOR_MODEL` | `qwen3.5:35b` | Via Ollama |
| `JUDGE_MODEL` | `llama3.1:8b` | Different family from generator |
| `TOP_K` | 5 | Pages per stream (10 total for hybrid) |
| `MAX_CHARS_PER_DOC` | 4000 | DeepSeek markdowns are verbose |
| `RERANK_TOP_K` | 100 | Dense candidates before reranking |

## Cache layout

```
data/processed/
    answers/{condition}/{subset}/{query_id}.json   ← generated answers (resume cache)
    judgments/{condition}/{subset}/{query_id}.json ← judgments (resume cache)
results_answers.csv                                ← pass@1 per condition/subset
```

Per-query JSON files are not committed to git (regeneratable). `results_answers.csv` is committed.
