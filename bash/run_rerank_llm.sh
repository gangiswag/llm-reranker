#!/bin/bash

# Set directories and model
DATA_DIR="${REPO_DIR}/datasets/"
OUTPUT_DIR="${REPO_DIR}/outputs/"

# Model configuration
RERANK_TYPE=${1:-"text"} # Default to text
CODE_PROMPT_TYPE=${2:-"docstring"} # Options: "docstring" or "github_issue" (only used when RERANK_TYPE=code)

if [ "$RERANK_TYPE" = "code" ]; then
    MODEL_IN_USE="cornstack/CodeRankLLM"
    # Code reranking doesn't support logits and alpha
    USE_LOGITS=0
    USE_ALPHA=0
else
    MODEL_IN_USE="rryisthebest/First_Model"
    # Text reranking configuration
    USE_LOGITS=1  # Whether to use FIRST single token logit decoding
    USE_ALPHA=1   # Whether to use Alphabetic Identifiers
fi

# List of datasets to rerank
if [ "$RERANK_TYPE" = "code" ]; then
    # Datasets suitable for code reranking
    DATASETS=('csn_ruby')  # 'javascript' 'go' 'php' 'ruby' 'java' 'python' 'cosqa'
else
    # Datasets for text reranking
    DATASETS=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'trec-covid'
fi

for DATASET in "${DATASETS[@]}"; do
    echo "Reranking dataset: ${DATASET} using ${RERANK_TYPE} reranking"
    
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
        --do_batched 1 \
        --rerank_type "${RERANK_TYPE}" \
        --code_prompt_type "${CODE_PROMPT_TYPE}"; then
        echo "Successfully reranked ${DATASET} with ${RERANK_TYPE} reranker"
    else
        echo "Failed to rerank ${DATASET} with ${RERANK_TYPE} reranker" >&2
        exit 1
    fi
done