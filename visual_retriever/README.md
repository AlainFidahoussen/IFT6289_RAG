# visual-retriever

ColEmbed (ColPali-style, Nemotron backbone) image-stream retrieval. Embeds ViDoRe v3 corpus page images and scores queries with MaxSim late interaction.

## Run

```bash
cd visual_retriever && uv sync && bash run_all.sh
```

Single subset dry run:

```bash
uv run visual_retriever/dataset.py --subset computer_science --lang english
uv run visual_retriever/predict.py --subset computer_science --lang english --save-rankings
```

All scripts are resume-safe: already-cached embeddings are skipped on re-run.

## Results (NDCG@10)

| Subset | NDCG@10 |
|---|---|
| computer_science | 78.04 |
| finance_en | 68.60 |
| pharmaceuticals | 67.23 |
| avg | 71.29 |
