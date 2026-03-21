#!/usr/bin/env bash
# Run ColEmbed dataset.py + predict.py for all 8 public ViDoRe v3 subsets.
# Must be run from inside visual_retriever/: cd visual_retriever && bash run_all.sh
set -euo pipefail

SUBSETS=(computer_science finance_en hr industrial pharmaceuticals physics energy finance_fr)
LANG="english"

for subset in "${SUBSETS[@]}"; do
    echo "====== $subset ======"
    uv run visual_retriever/dataset.py --subset "$subset" --lang "$LANG"
    uv run visual_retriever/predict.py --subset "$subset" --lang "$LANG"
done
