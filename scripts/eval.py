from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import csv
import os
import logging
import json
from argparse import ArgumentParser

def write_results_to_text(out_path, ndcg, recall):
    with open(out_path, 'w') as f:
        f.write(f"{ndcg}\n{recall}\n")

def eval_rank(output_path, data_dir, dataset, data_type, prefix="", eval_type="rank"):
    # Download datasets if necessary
    out_dir = os.path.join(data_dir, "datasets")
    data_path = os.path.join(out_dir, dataset)

    if data_type == "beir":
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)
    else:
        data_path = os.path.join(data_dir, dataset)

    # Load datasets
    split = "dev" if dataset == "msmarco" else "test"
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return

    # Set file name by evaluation type
    fname = "rank.tsv" if eval_type == "rank" else "rank_refit.tsv"

    # Load rank data
    model_output_path = os.path.join(output_path, dataset, fname)
    if not os.path.exists(model_output_path):
        print(f"Rank file not found: {model_output_path}")
        return

    results = {}
    try:
        with open(model_output_path, 'r') as rank_file:
            csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
            for row in csv_reader:
                qid, pid, score = row[0], row[1], float(row[2])
                if qid not in results:
                    results[qid] = {}
                results[qid][pid] = score
    except Exception as e:
        print(f"Error reading rank file {model_output_path}: {e}")
        return
    
    retriever = EvaluateRetrieval()

    # Evaluate retrieval using NDCG, MAP, Recall, Precision, and MRR
    print("Retriever evaluation")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 100, 125])
    mrr = retriever.evaluate_custom(qrels, results, [1, 3, 5, 10, 20, 100, 125], "mrr")
    
    if dataset == "trec-covid":
        recall = retriever.evaluate_custom(qrels, results, [1, 3, 5, 10, 20, 100, 125], "recall_cap")
    
    print(ndcg, recall)

def eval_rerank(output_path, data_dir, dataset, data_type, suffix=""):
    # Download datasets if necessary
    out_dir = os.path.join(data_dir, "datasets")
    data_path = os.path.join(out_dir, dataset)

    if data_type == "beir":
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)
    else:
        data_path = os.path.join(data_dir, dataset)

    split = "dev" if dataset == "msmarco" else "test"
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return

    model_output_path = os.path.join(output_path, dataset, f"rerank_100_{suffix}.json")
    if not os.path.exists(model_output_path):
        print(f"Rerank file not found: {model_output_path}")
        return

    try:
        with open(model_output_path, 'r') as json_file:
            results = json.load(json_file)
    except Exception as e:
        print(f"Error reading rerank file {model_output_path}: {e}")
        return

    retriever = EvaluateRetrieval()

    # Evaluate retrieval using NDCG, MAP, Recall, Precision, and MRR
    print("Retriever evaluation")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 100, 125])
    mrr = retriever.evaluate_custom(qrels, results, [1, 3, 5, 10, 20, 100, 125], "mrr")
    
    if dataset == "trec-covid":
        recall = retriever.evaluate_custom(qrels, results, [1, 3, 5, 10, 20, 100, 125], "recall_cap")
    
    print(ndcg)
    print(recall)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Name of the dataset to evaluate.")
    parser.add_argument('--output_path', required=True, help="Directory where the output files are stored.")
    parser.add_argument('--data_dir', required=True, help="Directory where datasets are stored or will be downloaded.")
    parser.add_argument('--suffix', default="", type=str, help="Suffix for the evaluation files (e.g., 'ce', 'logits_alpha').")
    parser.add_argument('--data_type', required=True, help="Type of the dataset, must be 'beir'.")
    parser.add_argument('--eval_type', required=True, help="Type of evaluation: 'rank', 'rerank', or 'rank_refit'.")
    args = parser.parse_args()

    assert args.data_type == "beir", "Currently, only 'beir' data_type is supported."
    assert args.eval_type in ["rank", "rerank", "rank_refit"], "Invalid eval_type. Must be 'rank', 'rerank', or 'rank_refit'."

    if args.eval_type in ["rank", "rank_refit"]:
        eval_rank(args.output_path, args.data_dir, args.dataset, args.data_type, args.suffix, args.eval_type)
    else:
        eval_rerank(args.output_path, args.data_dir, args.dataset, args.data_type, args.suffix)
