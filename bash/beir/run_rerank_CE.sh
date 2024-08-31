#!/bin/bash

# Set directories
DATA_DIR="${REPO_DIR}/datasets/beir/"
OUTPUT_DIR="${REPO_DIR}/outputs/beir/"

# List of datasets to rerank
DATASETS=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity'

# Iterate over datasets and rerank each one
for DATASET in "${DATASETS[@]}"; do  
    echo "Reranking dataset: ${DATASET}"
    
    # Execute the rerank script with error handling
    if python "${REPO_DIR}/scripts/rerank_CE.py" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        --data_type "beir" \
        --top_k 100; then
        echo "Successfully reranked ${DATASET} with CE reranker"
    else
        echo "Failed to rerank ${DATASET}" >&2
        exit 1
    fi
done
