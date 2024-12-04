import os
import json
from argparse import ArgumentParser
from beir.datasets.data_loader import GenericDataLoader
from utils.result import Result, ResultsLoader
from utils.llm_util import evaluate_results, get_results_to_eval, save_rerank_results, rerank_beir_outputs_llm

def rerank_beir_outputs(model, output_path, data_dir, dataset, data_type, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, context_size, rerank_type="text", code_prompt_type="docstring"):
    try:
        # Load dataset based on type
        if rerank_type == "code":
            data_path = os.path.join(data_dir, "code_datasets", dataset)
        else:  # text reranking
            data_path = os.path.join(data_dir, "beir", dataset)
            
        # Handle dataset loading
        split = "dev" if dataset == "msmarco" else "test"
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset} from {data_path}: {e}")
            return

        # Load converted retriever results
        try:
            if rerank_type == "code":
                results_output_path = os.path.join(output_path, "code_datasets", dataset, 'rank_100.json')
            else:
                results_output_path = os.path.join(output_path, "beir", dataset, 'rank_100.json')
                
            results_loader = ResultsLoader(results_output_path)
            results_to_rerank = results_loader.get_results(with_context=True)
        except Exception as e:
            print(f"Error loading results from {results_output_path}: {e}")
            return

        # Reranking
        try:
            reranked_results = rerank_beir_outputs_llm(
                model, results_to_rerank, use_logits=use_logits, use_alpha=use_alpha, 
                top_k=llm_top_k, window_size=window_size, step_size=step_size, 
                batched=batched, context_size=context_size,
                rerank_type=rerank_type, code_prompt_type=code_prompt_type
            )
        
            # Evaluate results
            converted_results = get_results_to_eval(reranked_results)
            
            if rerank_type == "code":
                mrr_at_k = evaluate_results(dataset, qrels, converted_results, rerank_type="code")
                print("\nMean Reciprocal Rank (MRR) at different cutoffs:")
                for k, mrr in mrr_at_k.items():
                    print(f"MRR@{k}: {mrr:.4f}")
            else:
                ndcg, _map, recall, precision = evaluate_results(dataset, qrels, converted_results)
                print(f"\nNDCG (Normalized Discounted Cumulative Gain):\n {ndcg}")
                print(f"\nRecall:\n {recall}\n")

            # Save rerank results to appropriate directory
            if rerank_type == "code":
                save_path = os.path.join(output_path, "code_datasets", dataset)
            else:
                save_path = os.path.join(output_path, "beir", dataset)
                
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            save_rerank_results(save_path, dataset, converted_results, llm_top_k, 
                              use_logits, use_alpha, is_llm_result=True)
            print(f"Reranked results saved successfully for dataset {dataset}")
            
        except Exception as e:
            print(f"Error during reranking process: {e}")
            raise
            
    except Exception as e:
        print(f"Unexpected error in rerank_beir_outputs: {e}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default="rryisthebest/First_Model")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--use_logits', default=0, type=int)
    parser.add_argument('--use_alpha', default=0, type=int)
    parser.add_argument('--context_size', default=32768, type=int)
    parser.add_argument('--llm_top_k', default=20, type=int)
    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--step_size', default=9, type=int)
    parser.add_argument('--do_batched', default=0, type=int)
    parser.add_argument('--rerank_type', type=str, default="text", choices=["text", "code"],
                      help="Whether to perform code or text reranking")
    parser.add_argument('--code_prompt_type', type=str, default="docstring", 
                      choices=["docstring", "github_issue"],
                      help="Type of code prompt to use (only applicable when rerank_type is 'code')")
    args = parser.parse_args()

    # Validate arguments
    if args.rerank_type == "text" and args.code_prompt_type != "docstring":
        print("Warning: code_prompt_type is ignored when rerank_type is 'text'")
    
    if args.rerank_type == "code" and (args.use_logits or args.use_alpha):
        print("Warning: Code reranking does not support logits or alpha mode. These will be disabled.")
        args.use_logits = 0
        args.use_alpha = 0

    rerank_beir_outputs(args.model, args.output_dir, args.data_dir, args.dataset, 
                       args.data_type, args.use_logits, args.use_alpha, args.llm_top_k,
                       args.window_size, args.step_size, args.do_batched, args.context_size,
                       args.rerank_type, args.code_prompt_type)