#!/usr/bin/env bash
# Run ColEmbed dataset.py + predict.py for all 8 public ViDoRe v3 subsets.
# Must be run from inside visual_retriever/: cd visual_retriever && bash run_all.sh
# Note: nuclear and telecom are private hold-out sets (not available on the Hub).
set -euo pipefail

rm -f results_colembed.csv

# subset, query_language
SUBSETS=(
    "computer_science english"
    "finance_en       english"
    "pharmaceuticals  english"
    "hr               english"
    "industrial       english"
    "physics          french"
    "energy           french"
    "finance_fr       french"
)

for entry in "${SUBSETS[@]}"; do
    subset=$(echo "$entry" | awk '{print $1}')
    lang=$(echo "$entry" | awk '{print $2}')
    echo "====== $subset ($lang) ======"
    uv run visual_retriever/dataset.py --subset "$subset" --lang "$lang"
    uv run visual_retriever/predict.py --subset "$subset" --lang "$lang"
done
