#!/bin/bash

# Check if input directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

input_dir=$1
output_dir="${REPO_DIR}/outputs/beir/"

# List of datasets to process
datasets=('trec-covid' 'dbpedia-entity') # 'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact' 'dbpedia-entity'

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}"

    python ${REPO_DIR}/scripts/prepare_distill.py \
        --output_path ${output_dir}/${dataset}/distill_input.pt \
        --rank_path ${output_dir}/${dataset}/rank.tsv \
        --psg_embs_dir ${input_dir}/${dataset}/original_corpus/ \
        --qry_embs_path ${input_dir}/${dataset}/original_query/qry.pt 

    if [ $? -ne 0 ]; then
        echo "Error processing dataset: ${dataset}"
        exit 1
    fi

    echo "Finished processing dataset: ${dataset}"
done
