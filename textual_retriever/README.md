# textual-retriever

Jina v4 text-stream retrieval with optional zerank-2 reranking. The markdown source is either the NeMo Retriever extraction built into the ViDoRe v3 dataset, or the DeepSeek-OCR-2 output from `textual_extraction/`. Swapping the source is the core experimental manipulation of the project.

## Run

```bash
cd textual_retriever && uv sync
bash run_all.sh                      # NeMo, no rerank
bash run_all.sh --rerank             # NeMo + zerank-2
bash run_all.sh --deepseek           # DeepSeek, no rerank
bash run_all.sh --deepseek --rerank  # DeepSeek + zerank-2
```

Single subset dry run:

```bash
uv run textual_retriever/dataset.py --subset computer_science --lang english [--source deepseek]
uv run textual_retriever/predict.py --subset computer_science --lang english [--source deepseek] [--rerank] [--save-rankings]
```

All scripts are resume-safe: already-cached embeddings are skipped on re-run.

## Results (NDCG@10)

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| NeMo, no rerank | 65.23 | 50.09 | 58.60 | 57.97 |
| NeMo + zerank-2 | 83.02 | 72.73 | 69.53 | **75.09** |
| DeepSeek, no rerank | 64.03 | 46.94 | 56.48 | 55.82 |
| DeepSeek + zerank-2 | 82.37 | 65.65 | 65.05 | 71.02 |
