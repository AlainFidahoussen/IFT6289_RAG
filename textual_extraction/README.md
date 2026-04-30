# textual-extraction

Runs DeepSeek-OCR-2 over ViDoRe v3 corpus page images and saves the extracted markdown. Output is consumed by `textual_retriever/` as the parser under comparison (against NeMo Retriever, which is built into the dataset).

## Run

```bash
cd textual_extraction && uv sync && bash run_all.sh
```

Single subset dry run:

```bash
uv run textual_extraction/dataset.py --subset computer_science --lang english
```

Already-processed pages are skipped on re-run.
