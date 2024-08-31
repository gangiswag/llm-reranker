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

def evaluate_results(dataset, qrels, rerank_results):
    retriever = EvaluateRetrieval()
    metrics_to_evaluate = [1, 3, 5, 10, 20, 100]
    
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
    rerank_path = os.path.join(output_path, dataset, f"rerank_{top_k}{suffix}.json")
    
    print(f"Saved to: {rerank_path}")
    with open(rerank_path, "w") as f:
        json.dump(results, f, indent=4)

def rerank_beir_outputs_llm(model, results_for_rerank, use_logits, use_alpha, top_k, window_size, step_size, batched, context_size):
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
        system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
        batched=batched
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
