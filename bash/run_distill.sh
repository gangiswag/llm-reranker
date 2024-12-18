#!/bin/bash

# Check if output directory exists
mkdir -p "${REPO_DIR}/outputs/beir"
output_dir="${REPO_DIR}/outputs/beir/"
data_dir="${REPO_DIR}/datasets/beir/"

# Configuration flags
use_logits=1  # Whether to use FIRST single token logit decoding
use_alpha=1   # Whether to use Alphabetic Identifiers

# List of datasets to process
datasets=('trec-covid' 'dbpedia-entity') #'climate-fever' 'fever' 'hotpotqa' 'msmarco' 'nfcorpus' 'nq' 'fiqa' 'scidocs' 'scifact'

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}"

    python ${REPO_DIR}/scripts/distill.py \
        --inp_path ${output_dir}/${dataset}/distill_input.pt \
        --rerank_path ${output_dir}/${dataset} \
        --output_path ${output_dir}/${dataset}/qry_refit.pt \
        --ce_top_k 100 \
        --llm_top_k 100 \
        --use_logits ${use_logits} \
        --use_alpha ${use_alpha} \
        --loss_path ${output_dir}/${dataset} \
        --llm_loss ranknet

    if [ $? -ne 0 ]; then
        echo "Error processing dataset: ${dataset}"
        exit 1
    fi

    echo "Finished processing dataset: ${dataset}"
done

echo "All datasets processed successfully."
