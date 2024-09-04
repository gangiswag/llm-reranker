#!/bin/bash

# Check if eval_type argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <eval_type>"
    exit 1
fi

EVAL_TYPE=$1
DATA_DIR="${REPO_DIR}/datasets/beir/"
OUTPUT_DIR="${REPO_DIR}/outputs/beir/"

# List of datasets to process
DATASETS=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity'

# Iterate over datasets and process each one
for DATASET in "${DATASETS[@]}"; do
    echo "Evaluating dataset: ${DATASET}"
    
    # suffix: ce -> cross encoder reranker | llm_FIRST_alpha -> FIRST Model
    if python "${REPO_DIR}/scripts/eval.py" \
        --dataset "${DATASET}" \
        --output_path "${OUTPUT_DIR}" \
        --data_type "beir" \
        --suffix "llm_FIRST_alpha" \
        --eval_type "${EVAL_TYPE}" \
        --data_dir "${DATA_DIR}"; then
        echo "Successfully evaluated ${DATASET}"
    else
        echo "Failed to evaluate ${DATASET}" >&2
        exit 1
    fi
done