#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <eval_type> [rerank_type]"
    exit 1
fi

EVAL_TYPE=$1
RERANK_TYPE=${2:-"text"}  # Default to text if not specified

DATA_DIR="${REPO_DIR}/datasets/"
OUTPUT_DIR="${REPO_DIR}/outputs/"

if [ "$RERANK_TYPE" = "code" ]; then
    # Code datasets to process
    DATASETS=('csn_go')  # 'csn_go' 'csn_java' 'csn_python' 'csn_javascript' 'csn_php' 'csn_ruby' 'cosqa'
    DATA_TYPE="codedataset"
else
    # BEIR datasets to process
    DATASETS=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity'
    DATA_TYPE="beir"
fi

for DATASET in "${DATASETS[@]}"; do
    echo "Evaluating dataset: ${DATASET} (${RERANK_TYPE} reranking)"
    
    # suffix: ce -> cross encoder reranker | llm_FIRST_alpha -> FIRST Model
    if python "${REPO_DIR}/scripts/eval.py" \
        --dataset "${DATASET}" \
        --output_path "${OUTPUT_DIR}" \
        --data_type "${DATA_TYPE}" \
        --suffix "llm_FIRST_alpha" \
        --eval_type "${EVAL_TYPE}" \
        --data_dir "${DATA_DIR}" \
        --rerank_type "${RERANK_TYPE}"; then
        echo "Successfully evaluated ${DATASET}"
    else
        echo "Failed to evaluate ${DATASET}" >&2
        exit 1
    fi
done