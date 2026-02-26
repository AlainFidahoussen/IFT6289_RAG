# Two projects (uv)

This repo contains two **separate uv projects**, each with its own `pyproject.toml` and lockfile:

| Project            | Path               | Use case                    | Transformers |
|--------------------|--------------------|-----------------------------|--------------|
| **visual-retriever** | `visual_retriever/` | ColEmbed, ViDoRe visual     | 5.x          |
| **textual-retriever** | `textual_retriever/` | Jina v4, ViDoRe text-only    | 4.x          |

## Install and run

**Visual retriever (ColEmbed):**
```bash
cd visual_retriever
uv sync
uv run python -m visual_retriever.model
```

**Textual retriever (Jina v4):**
```bash
cd textual_retriever
uv sync
uv run python -m textual_retriever.model
```

No shared venv: each project has its own dependencies and `uv sync` in that directory is enough.
