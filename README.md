# ift6289_rag

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Text Extraction Quality in Vision-Text Hybrid Retrieval: Controlled End-to-End QA Evaluation on Visually Rich Documents


## Two projects (uv)

This repo contains two **separate uv projects**, each with its own `pyproject.toml` and lockfile:

| Project               | Path                 | Use case                  | 
|-----------------------|----------------------|---------------------------|
| **visual-retriever**  | `visual_retriever/`  | ColEmbed, ViDoRe visual   |
| **textual-retriever** | `textual_retriever/` | Jina v4, ViDoRe text-only |

## Install and run

**Visual retriever (ColEmbed):**
```bash
cd visual_retriever
uv sync
uv run visual_retriever/dataset.py # To build the data by pre-computing and saving embedding
uv run visual_retriever/predict.py # To compute the NDCG@10
```

**Textual retriever (Jina v4):**
```bash
cd textual_retriever
uv sync
uv run textual_retriever/dataset.py # To build the data by pre-computing and saving embedding
uv run textual_retriever/predict.py # To compute the NDCG@10
```

## Project Organization

```
├── LICENSE              <- Open-source license
├── Makefile             <- Convenience commands (e.g. make data, make train)
├── README.md            <- This file
│
├── textual_retriever    <- Text retrieval project (Jina v4)
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── data/            <- Dataset and caches (gitignored)
│   │   └── processed
│   ├── notebooks/
│   └── textual_retriever <- Python package
│       ├── __init__.py
│       ├── config.py    <- Paths and settings
│       ├── dataset.py   <- Data loading
│       ├── features.py  <- Feature extraction / embedding cache
│       ├── model.py     <- Model definition
│       ├── predict.py   <- Inference and evaluation
│       └── utils.py     <- Helpers (e.g. NDCG)
│
└── visual_retriever     <- Visual retrieval project (ColEmbed)
    ├── pyproject.toml
    ├── uv.lock
    ├── data/            <- Dataset and caches (gitignored)
    │   └── processed
    ├── notebooks/
    └── visual_retriever  <- Python package
        ├── __init__.py
        ├── config.py
        ├── dataset.py
        ├── features.py
        ├── model.py
        ├── predict.py
        └── utils.py
```

--------

