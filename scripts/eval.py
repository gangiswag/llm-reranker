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

def eval_rank(output_path, data_dir, dataset, data_type, prefix="", eval_type="rank", rerank_type="text"):
    try:
        # Load datasets based on type
        if rerank_type == "code":
            data_path = os.path.join(data_dir, "code_datasets", dataset)
            split = "test"
        else:  # text reranking
            data_path = os.path.join(data_dir, "beir", dataset)
            split = "dev" if dataset == "msmarco" else "test"

        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset} from {data_path}: {e}")
            return

        # Set file name by evaluation type
        fname = "rank.tsv" if eval_type == "rank" else "rank_refit.tsv"

        # Load rank data from appropriate directory
        if rerank_type == "code":
            model_output_path = os.path.join(output_path, "code_datasets", dataset, fname)
        else:
            model_output_path = os.path.join(output_path, "beir", dataset, fname)

        if not os.path.exists(model_output_path):
            print(f"Rank file not found: {model_output_path}")
            return

        results = {}
        try:
            with open(model_output_path, 'r') as rank_file:
                csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
                if rerank_type == "code":
                    next(csv_reader)  # Skip header for code files
                for row in csv_reader:
                    qid, pid, score = row[0], row[1], float(row[2])
                    if qid not in results:
                        results[qid] = {}
                    results[qid][pid] = score
        except Exception as e:
            print(f"Error reading rank file {model_output_path}: {e}")
            return
        
        retriever = EvaluateRetrieval()
        metrics_to_evaluate = [1, 3, 5, 10, 20, 100, 125]

        # Evaluate based on rerank type
        if rerank_type == "code":
            mrr = retriever.evaluate_custom(qrels, results, metrics_to_evaluate, "mrr")
            print(f"MRR@{metrics_to_evaluate}: {mrr}")
        else:
            # Standard text reranking evaluation
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, metrics_to_evaluate)
            if dataset == "trec-covid":
                recall = retriever.evaluate_custom(qrels, results, metrics_to_evaluate, "recall_cap")
            print(ndcg, recall)

    except Exception as e:
        print(f"Error in eval_rank: {e}")
        raise

def eval_rerank(output_path, data_dir, dataset, data_type, suffix="", rerank_type="text"):
    try:
        # Load datasets based on type
        if rerank_type == "code":
            data_path = os.path.join(data_dir, "code_datasets", dataset)
            split = "test"
        else:  # text reranking
            data_path = os.path.join(data_dir, "beir", dataset)
            split = "dev" if dataset == "msmarco" else "test"

        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset} from {data_path}: {e}")
            return

        # Load rerank results from appropriate directory
        if rerank_type == "code":
            model_output_path = os.path.join(output_path, "code_datasets", dataset, f"rerank_100_{suffix}.json")
        else:
            model_output_path = os.path.join(output_path, "beir", dataset, f"rerank_100_{suffix}.json")

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
        metrics_to_evaluate = [1, 3, 5, 10, 20, 100]

        if rerank_type == "code":
            # For code reranking, focus on MRR and exact matches
            mrr = retriever.evaluate_custom(qrels, results, metrics_to_evaluate, "mrr")
            print(f"MRR@{metrics_to_evaluate}: {mrr}")
        else:
            # Standard text reranking evaluation
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, metrics_to_evaluate)
            mrr = retriever.evaluate_custom(qrels, results, metrics_to_evaluate, "mrr")
            print(ndcg, recall)

    except Exception as e:
        print(f"Error in eval_rerank: {e}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Name of the dataset to evaluate.")
    parser.add_argument('--output_path', required=True, help="Directory where the output files are stored.")
    parser.add_argument('--data_dir', required=True, help="Directory where datasets are stored or will be downloaded.")
    parser.add_argument('--suffix', default="", type=str, help="Suffix for the evaluation files (e.g., 'ce', 'logits_alpha').")
    parser.add_argument('--data_type', required=True, help="Type of the dataset, must be 'beir' or 'codedataset'.")
    parser.add_argument('--eval_type', required=True, help="Type of evaluation: 'rank', 'rerank', or 'rank_refit'.")
    parser.add_argument('--rerank_type', type=str, default="text", choices=["text", "code"],
                      help="Whether to evaluate code or text reranking results")
    args = parser.parse_args()

    assert args.data_type in ["beir", "codedataset"], "Invalid data_type. Must be 'beir' or 'codedataset'."
    assert args.eval_type in ["rank", "rerank", "rank_refit"], "Invalid eval_type. Must be 'rank', 'rerank', or 'rank_refit'."

    if args.data_type == "codedataset" and args.rerank_type != "code":
        print("Warning: codedataset data_type implies code reranking. Setting rerank_type to 'code'")
        args.rerank_type = "code"

    if args.eval_type in ["rank", "rank_refit"]:
        eval_rank(args.output_path, args.data_dir, args.dataset, args.data_type, 
                 args.suffix, args.eval_type, args.rerank_type)
    else:
        eval_rerank(args.output_path, args.data_dir, args.dataset, args.data_type, 
                   args.suffix, args.rerank_type)
