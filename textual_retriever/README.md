# textual-retriever

ViDoRe **textual** retriever (Jina v4). Uses **transformers 4.x** (separate from visual-retriever).

## Install

```bash
uv sync
```

## Create dataset

```bash
# Create dataset
uv run textual_retriever/dataset.py
```

## Compute metric NCDG@10

```bash
# Create dataset
uv run textual_retriever/predict.py
```
```
