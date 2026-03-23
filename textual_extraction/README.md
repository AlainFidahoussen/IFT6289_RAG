# textual-extraction

OCR/parsing subproject for the ift6289_rag pipeline. Runs **DeepSeek-OCR-2** over ViDoRe v3 corpus page images and saves the extracted markdown to disk, which is then consumed by the `textual-retriever` subproject.

This is the **variable under study**: swapping this extractor (DeepSeek-OCR-2 vs. NeMo Retriever extraction, used in the ViDoRe v3 paper) is the core experimental manipulation of the project.

## Role in the pipeline

```
ViDoRe v3 corpus images
        │
        ▼
 textual_extraction/dataset.py   ← you are here
        │  DeepSeek-OCR-2
        │  saves <corpus_id>/result.mmd per page
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

## Run all subsets

```bash
bash run_all.sh
```

This runs DeepSeek-OCR-2 on the three active English subsets. Already-processed documents are skipped so the script is safe to resume after interruptions.

## Run a single subset

```bash
uv run textual_extraction/dataset.py --subset computer_science --lang english
```

## Active subsets

| Subset | Language |
|---|---|
| computer_science | english |
| finance_en | english |
| pharmaceuticals | english |

## Cache layout

Results are stored under `data/processed/` as one directory per corpus document:

```
deepseek_cache_markdowns_{subset}_{lang}/
    {corpus_id}/
        result.mmd          ← extracted markdown text
        result_with_boxes.jpg
        images/
```
