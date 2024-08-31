data_dir=${REPO_DIR}/datasets/beir/
output_dir=${REPO_DIR}/outputs/beir/

# List of datasets to process
datasets=('trec-covid') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity' 'trec-covid'

# Iterate over datasets and process each one
for datasets in "${datasets[@]}"; do
    echo "Processing dataset: ${datasets}"
    
    # Execute the conversion script with error handling
    if python "${REPO_DIR}/scripts/convert_results.py" \
        --dataset "${datasets}" \
        --output_dir "${output_dir}" \
        --data_type "beir" \
        --data_dir "${data_dir}" \
        --top_k 100; then
        echo "Successfully processed ${datasets}"
    else
        echo "Failed to process ${datasets}" >&2
        exit 1
    fi
done