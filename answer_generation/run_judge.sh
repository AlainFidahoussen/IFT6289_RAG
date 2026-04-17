#!/usr/bin/env bash
# Judge generated answers for all three ViDoRe v3 subsets under a given condition.
# Must be run from inside answer_generation/: cd answer_generation && bash run_judge.sh --condition <cond>
#
# Run AFTER run_all.sh has generated answers.
#
# Examples:
#   bash run_judge.sh --condition hybrid_nemo
set -euo pipefail

CONDITION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)
            CONDITION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONDITION" ]; then
    echo "ERROR: --condition is required."
    echo "Usage: bash run_judge.sh --condition <condition>"
    exit 1
fi

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
    echo "====== $subset ($lang) [condition=$CONDITION] ======"

    if ! uv run answer_generation/judge.py --subset "$subset" --lang "$lang" --condition "$CONDITION"; then
        echo "ERROR: $subset ($lang) judge step failed, skipping."
        FAILED+=("$subset ($lang)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "====== FAILED SUBSETS ======"
    for f in "${FAILED[@]}"; do echo "  $f"; done
fi
