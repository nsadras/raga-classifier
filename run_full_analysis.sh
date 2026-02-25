#!/bin/bash
set -e

# Generate timestamp
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M")
RESULTS_DIR="results-${TIMESTAMP}"
echo "Starting full analysis. Output directory: ${RESULTS_DIR}"

mkdir -p ${RESULTS_DIR}

# 1. Run MERT analysis (Full Dataset -> all segments)
echo "----------------------------------------------------------------"
echo "Running MERT Analysis..."
uv run python main.py \
    --task umap \
    --model_type mert \
    --strategy smart \
    --min_melodic_score -4.0 \
    --batch_size 16 \
    --cookies_file cookies.txt \
    --results_dir ${RESULTS_DIR}

# 2. Run CultureMERT analysis (Full Dataset)
echo "----------------------------------------------------------------"
echo "Running CultureMERT Analysis..."
uv run python main.py \
    --task umap \
    --model_type culturemert \
    --strategy smart \
    --min_melodic_score -4.0 \
    --batch_size 16 \
    --cookies_file cookies.txt \
    --results_dir ${RESULTS_DIR}

echo "----------------------------------------------------------------"
echo "Analysis Complete."
echo "Results available in: ${RESULTS_DIR}"
