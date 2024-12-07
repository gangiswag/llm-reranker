#!/bin/bash

# Set directories and model
DATA_DIR="${REPO_DIR}/datasets/beir/"
OUTPUT_DIR="${REPO_DIR}/outputs/beir/"
MODEL_IN_USE="rryisthebest/First_Model"

# Configuration flags
USE_LOGITS=1  # Whether to use FIRST single token logit decoding
USE_ALPHA=1   # Whether to use Alphabetic Identifiers

# List of datasets to rerank
DATASETS=('dbpedia-entity') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'trec-covid'

# Iterate over datasets and rerank each one
for DATASET in "${DATASETS[@]}"; do
    echo "Reranking dataset: ${DATASET}"
    
    if python "${REPO_DIR}/scripts/rerank_llm.py" \
        --model "${MODEL_IN_USE}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_type "beir" \
        --data_dir "${DATA_DIR}" \
        --use_logits "${USE_LOGITS}" \
        --use_alpha "${USE_ALPHA}" \
        --llm_top_k 100 \
        --window_size 20 \
        --step_size 10 \
        --do_batched 1; then
        echo "Successfully reranked ${DATASET} with LLM reranker"
    else
        echo "Failed to rerank ${DATASET} with LLM reranker" >&2
        exit 1
    fi
done