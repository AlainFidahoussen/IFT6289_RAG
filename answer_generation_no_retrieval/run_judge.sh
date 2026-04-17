#!/usr/bin/env bash
# Judge closed-book answers for all three ViDoRe v3 subsets.
# Must be run from inside answer_generation_no_retrieval/:
#   cd answer_generation_no_retrieval && bash run_judge.sh
#
# Run AFTER run_all.sh has generated answers.
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
    echo "====== $subset ($lang) [closed_book] ======"

    if ! uv run answer_generation_no_retrieval/judge.py --subset "$subset" --lang "$lang"; then
        echo "ERROR: $subset ($lang) judge step failed, skipping."
        FAILED+=("$subset ($lang)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "====== FAILED SUBSETS ======"
    for f in "${FAILED[@]}"; do echo "  $f"; done
fi
