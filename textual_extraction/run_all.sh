#!/usr/bin/env bash
# Run DeepSeek-OCR-2 extraction for the three active ViDoRe v3 subsets.
# Must be run from inside textual_extraction/: cd textual_extraction && bash run_all.sh
set -euo pipefail

SUBSETS=(
    "computer_science english"
    "finance_en       english"
    "pharmaceuticals  english"
    "finance_fr       french"
    "physics          french"
)

FAILED=()
for entry in "${SUBSETS[@]}"; do
    subset=$(echo "$entry" | awk '{print $1}')
    lang=$(echo "$entry" | awk '{print $2}')
    echo "====== $subset ($lang) ======"
    if ! uv run textual_extraction/dataset.py --subset "$subset" --lang "$lang"; then
        echo "ERROR: $subset ($lang) failed, skipping."
        FAILED+=("$subset ($lang)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "====== FAILED SUBSETS ======"
    for f in "${FAILED[@]}"; do echo "  $f"; done
fi
