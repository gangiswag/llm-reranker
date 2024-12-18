import csv
import os
import logging
import json
import math
from beir.retrieval.evaluation import EvaluateRetrieval
from utils.result import Result, ResultsLoader
from utils.rankllm import PromptMode, RankLLM
from utils.reranker import Reranker
from utils.rank_listwise_os_llm import RankListwiseOSLLM

def evaluate_results(dataset, qrels, rerank_results, rerank_type="text"):
    """
    Evaluate reranking results for both text and code datasets
    
    Args:
        dataset: Name of the dataset
        qrels: Ground truth relevance judgments
        rerank_results: Results to evaluate
        rerank_type: Either "text" or "code" reranking
    
    Returns:
        For text reranking: (ndcg, _map, recall, precision)
        For code reranking: Dictionary of MRR values at different cutoffs
    """
    metrics_to_evaluate = [1, 3, 5, 10, 20, 100]
    
    if rerank_type == "code":
        mrr_at_k = {}
        
        for k in metrics_to_evaluate:
            mrr_sum = 0.0
            num_queries = 0
            
            for qid in qrels:
                if qid in rerank_results:
                    sorted_docs = sorted(rerank_results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
                    
                    for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                        if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                            mrr_sum += 1.0 / rank
                            break
                    num_queries += 1
            
            mrr = mrr_sum / num_queries if num_queries > 0 else 0.0
            mrr_at_k[k] = mrr
            
        return mrr_at_k
    else:  # text reranking
        retriever = EvaluateRetrieval()
        
        ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, metrics_to_evaluate)
        
        if dataset == "trec-covid":
            recall_cap_metrics = metrics_to_evaluate + [125]
            recall = retriever.evaluate_custom(qrels, rerank_results, recall_cap_metrics, metric="recall_cap")
        
        return ndcg, _map, recall, precision

def get_results_to_eval(results):
    eval_results = {}
    
    for result in results:
        hits = result.hits
        qid = hits[0]['qid']
        eval_results[qid] = {hit['docid']: hit['score'] for hit in hits}
    
    return eval_results

def save_rerank_results(output_path, dataset, results, top_k, use_logits=False, use_alpha=False, is_llm_result=False):
    suffix_parts = []
    
    if is_llm_result:
        suffix_parts.append("_llm")
        suffix_parts.append("_FIRST" if use_logits else "_gen")
        suffix_parts.append("_alpha" if use_alpha else "_num")
    else:
        suffix_parts.append("_ce")
    
    suffix = "".join(suffix_parts)
    if output_path.endswith(dataset):
        rerank_path = os.path.join(output_path, f"rerank_{top_k}{suffix}.json")
    else:
        rerank_path = os.path.join(output_path, f"rerank_{top_k}{suffix}.json")
    
    os.makedirs(os.path.dirname(rerank_path), exist_ok=True)
    
    print(f"Saved to: {rerank_path}")
    with open(rerank_path, "w") as f:
        json.dump(results, f, indent=4)

def rerank_beir_outputs_llm(model, results_for_rerank, use_logits, use_alpha, top_k, window_size, step_size, batched, context_size, rerank_type="text", code_prompt_type="docstring"):
    """
    Rerank outputs using either text or code reranking
    
    Args:
        rerank_type (str): Whether to perform "text" or "code" reranking
        code_prompt_type (str): For code reranking, whether to use "docstring" or "github_issue" prompts
    """
    # Validate parameters for code reranking
    if rerank_type == "code":
        if use_logits or use_alpha:
            print("Warning: Code reranking does not support logits or alpha mode. These will be disabled.")
            use_logits = False
            use_alpha = False

    # Select appropriate system message based on rerank type and prompt type
    if rerank_type == "code":
        if code_prompt_type == "docstring":
            system_message = "You are CodeRanker, an intelligent code reviewer that can analyze doc strings and rank code snippets based on their relevance to the doc string."
        elif code_prompt_type == "github_issue": 
            system_message = "You are CodeRanker, an intelligent code reviewer that can analyze GitHub issues and rank code functions based on their relevance to contain the faults causing the GitHub issue."
        else:
            raise ValueError(f"Invalid code_prompt_type: {code_prompt_type}")
    else:  # text reranking
        system_message = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"

    # Initialize the ranking model
    agent = RankListwiseOSLLM(
        model=model,
        context_size=context_size,
        prompt_mode=PromptMode.RANK_GPT,
        num_few_shot_examples=0,
        device="cuda",
        num_gpus=1,
        variable_passages=True,
        window_size=window_size,
        system_message=system_message,
        batched=batched,
        rerank_type=rerank_type,
        code_prompt_type=code_prompt_type
    )

    # Perform reranking
    reranker = Reranker(agent=agent)
    reranked_results = reranker.rerank(
        retrieved_results=results_for_rerank,
        use_logits=use_logits,
        use_alpha=use_alpha,
        rank_start=0,
        rank_end=top_k,
        window_size=window_size,
        step=step_size,
        logging=False,
        batched=batched
    )

    for result in reranked_results:
        for rank, hit in enumerate(result.hits, start=1):
            hit['rank'] = rank

    return reranked_results
