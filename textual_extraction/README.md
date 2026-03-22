# textual-extraction

OCR/parsing subproject for the ift6289_rag pipeline. Runs **DeepSeek-OCR-2** over ViDoRe v3 corpus page images and saves the extracted markdown to disk, which is then consumed by the `textual-retriever` subproject.

This is the **variable under study**: swapping this extractor (e.g. DeepSeek-OCR-2 vs. NeMo Retriever extraction) is the core experimental manipulation of the project.

## Role in the pipeline

```
ViDoRe v3 corpus images
        │
        ▼
 textual_extraction/dataset.py   ← you are here
        │  DeepSeek-OCR-2
        │  saves {corpus_id}.md per page
        ▼
 textual_retriever/dataset.py
        │  Jina v4 embeds the markdown
        ▼
 textual_retriever/predict.py
        │  NDCG@10
```

## Install

```bash
cd textual_extraction
uv sync
```

## Run

```bash
uv run textual_extraction/dataset.py
```

This iterates over all corpus images for the configured subset, runs DeepSeek-OCR-2 inference on each, and saves the markdown output to:

```
data/processed/<cache_dir>/{corpus_id}
```

Already-processed files are skipped on re-runs, so the script is safe to resume after interruptions.

## Configuration

Edit `textual_extraction/config.py` to change the active subset:

```python
VIDORE_SUBSET = "computer_science"   # or "physics", "finance", etc.
VIDORE_LANG   = "english"
```
