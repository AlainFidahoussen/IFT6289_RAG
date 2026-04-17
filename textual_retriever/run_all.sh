#!/usr/bin/env bash
# Embed + evaluate Jina v4 for all three ViDoRe v3 subsets.
# Must be run from inside textual_retriever/: cd textual_retriever && bash run_all.sh [options]
#
# Options:
#   --deepseek   Use DeepSeek-OCR-2 extracted markdown (default: NeMo built-in)
#   --rerank     Apply zerank-2 reranker after dense retrieval (skips dataset.py embedding step)
#
# Examples:
#   bash run_all.sh                        # NeMo, no rerank  → results_jina.csv
#   bash run_all.sh --rerank               # NeMo + zerank-2  → results_jina_reranked.csv
#   bash run_all.sh --deepseek             # DeepSeek         → results_jina_deepseek.csv
#   bash run_all.sh --deepseek --rerank    # DeepSeek+zerank-2→ results_jina_reranked_deepseek.csv
set -euo pipefail

SOURCE="nemo"
RERANK=false

for arg in "$@"; do
    case "$arg" in
        --deepseek) SOURCE="deepseek" ;;
        --rerank)   RERANK=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

SUBSETS=(
    "computer_science english"
    "finance_en       english"
    "pharmaceuticals  english"
    "finance_fr       french"
    "physics          french"
)

SOURCE_SUFFIX=""
[ "$SOURCE" != "nemo" ] && SOURCE_SUFFIX="_${SOURCE}"

if $RERANK; then
    RESULTS_FILE="results_jina_reranked${SOURCE_SUFFIX}.csv"
else
    RESULTS_FILE="results_jina${SOURCE_SUFFIX}.csv"
fi

rm -f "$RESULTS_FILE"

FAILED=()
for entry in "${SUBSETS[@]}"; do
    subset=$(echo "$entry" | awk '{print $1}')
    lang=$(echo "$entry" | awk '{print $2}')
    echo "====== $subset ($lang) [source=$SOURCE rerank=$RERANK] ======"

    if ! $RERANK; then
        if ! uv run textual_retriever/dataset.py --subset "$subset" --lang "$lang" --source "$SOURCE"; then
            echo "ERROR: $subset ($lang) dataset step failed, skipping."
            FAILED+=("$subset ($lang)")
            continue
        fi
    fi

    PREDICT_ARGS=(--subset "$subset" --lang "$lang" --source "$SOURCE" --save-rankings)
    $RERANK && PREDICT_ARGS+=(--rerank)

    if ! uv run textual_retriever/predict.py "${PREDICT_ARGS[@]}"; then
        echo "ERROR: $subset ($lang) predict step failed, skipping."
        FAILED+=("$subset ($lang)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "====== FAILED SUBSETS ======"
    for f in "${FAILED[@]}"; do echo "  $f"; done
fi
