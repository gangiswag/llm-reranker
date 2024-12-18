#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

input_dir=$1

# Check if output directory exists
mkdir -p "${REPO_DIR}/outputs/beir"

# List of datasets to process
datasets=('trec-covid' 'dbpedia-entity') #'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact'

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}"

    dataset_output_dir="${REPO_DIR}/outputs/beir/${dataset}"
    mkdir -p "$dataset_output_dir"

    python -m tevatron.faiss_retriever \
        --query_reps "${dataset_output_dir}/qry_refit.pt" \
        --passage_reps "${input_dir}/${dataset}/original_corpus/*.pt" \
        --depth 1000 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to "${dataset_output_dir}/rank_refit.tsv"

    if [ $? -ne 0 ]; then
        echo "Error processing dataset: ${dataset}"
        exit 1
    fi

    echo "Finished processing dataset: ${dataset}"
done

echo "All datasets processed successfully."
