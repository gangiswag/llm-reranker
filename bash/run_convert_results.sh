#!/bin/bash

data_dir=${REPO_DIR}/datasets/
output_dir=${REPO_DIR}/outputs/

RERANK_TYPE=${1:-"text"}  # Default to text if no argument provided

if [ "$RERANK_TYPE" = "code" ]; then
    # Code datasets to process
    datasets=('csn_go')  # 'csn_go' 'csn_java' 'csn_python' 'csn_javascript' 'csn_php' 'csn_ruby' 'cosqa'
    data_type="codedataset"
else
    # BEIR datasets to process
    datasets=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity'
    data_type="beir"
fi

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset} (${RERANK_TYPE} reranking)"
    
    if python "${REPO_DIR}/scripts/convert_results.py" \
        --dataset "${dataset}" \
        --output_dir "${output_dir}" \
        --data_type "${data_type}" \
        --data_dir "${data_dir}" \
        --top_k 100 \
        --rerank_type "${RERANK_TYPE}"; then
        echo "Successfully processed ${dataset}"
    else
        echo "Failed to process ${dataset}" >&2
        exit 1
    fi
done