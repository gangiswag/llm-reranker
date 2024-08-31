import os
import json
from argparse import ArgumentParser
from beir.datasets.data_loader import GenericDataLoader
from utils.result import Result, ResultsLoader
from utils.llm_util import evaluate_results, get_results_to_eval, save_rerank_results, rerank_beir_outputs_llm

def rerank_beir_outputs(model, output_path, data_dir, dataset, data_type, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, context_size):
    try:
        # Load BEIR dataset
        data_path = os.path.join(data_dir, "datasets", dataset)
        split = "dev" if dataset == "msmarco" else "test"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return

    try:
        # Load converted retriever results
        results_output_path = os.path.join(output_path, dataset, 'rank_100.json')
        results_loader = ResultsLoader(results_output_path)
        results_to_rerank = results_loader.get_results(with_context=True)
    except Exception as e:
        print(f"Error loading results from {results_output_path}: {e}")
        return


    try:
        # Reranking
        reranked_results = rerank_beir_outputs_llm(
            model, results_to_rerank, use_logits=use_logits, use_alpha=use_alpha, 
            top_k=llm_top_k, window_size=window_size, step_size=step_size, 
            batched=batched, context_size=context_size
        )
    
        # rerank evaluation
        converted_results = get_results_to_eval(reranked_results)
        ndcg, _map, recall, precision = evaluate_results(dataset, qrels, converted_results)
        print(f"\nNDCG (Normalized Discounted Cumulative Gain):\n {ndcg}")
        print(f"\nRecall:\n {recall}\n")

        # save rerank results
        save_rerank_results(output_path, dataset, converted_results, llm_top_k, use_logits, use_alpha, is_llm_result=True)
        print(f"Reranked results saved successfully for dataset {dataset}")
    except Exception as e:
        print(f"Error during reranking process: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default="rryisthebest/First_Model")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--use_logits', default=0, type=int)
    parser.add_argument('--use_alpha', default=0, type=int)
    parser.add_argument('--context_size', default=4096, type=int)
    parser.add_argument('--llm_top_k', default=20, type=int)
    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--step_size', default=9, type=int)
    parser.add_argument('--do_batched', default=0, type=int)
    args = parser.parse_args()

    assert args.data_type == "beir"

    rerank_beir_outputs(args.model, args.output_dir, args.data_dir, args.dataset, args.data_type, args.use_logits, args.use_alpha, args.llm_top_k, args.window_size, args.step_size, args.do_batched, args.context_size)