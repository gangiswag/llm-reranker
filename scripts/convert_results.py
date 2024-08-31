from beir.datasets.data_loader import GenericDataLoader
import csv
import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from utils.result import Result, ResultsWriter

def convert_results(output_path, data_dir, dataset, data_type, top_k):
    # Load datasets
    print(f"Loading {dataset} dataset")
    out_dir = os.path.join(data_dir, "datasets")
    data_path = os.path.join(out_dir, dataset)

    try:
        if dataset == "msmarco":
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
        else:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return

    print("Loading rank data")
    dataset_output_path = os.path.join(output_path, dataset)
    rank_path = os.path.join(dataset_output_path, "rank.tsv")

    # Ensure the rank file exists
    if not os.path.exists(rank_path):
        print(f"Rank file not found: {rank_path}")
        return

    results = {}
    try:
        with open(rank_path, 'r') as rank_file:
            csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
            for row in csv_reader:
                qid = str(row[0])
                pid = str(row[1])
                score = float(row[2])
                if qid not in results:
                    results[qid] = {}
                results[qid][pid] = score
    except Exception as e:
        print(f"Error reading rank file {rank_path}: {e}")
        return
    
    print("Converting to LLM results")
    results_to_rerank = to_llm_results(results, queries, corpus, top_k)

    # Ensure output directory exists
    os.makedirs(dataset_output_path, exist_ok=True)
    
    results_output_path = os.path.join(dataset_output_path, f'rank_{top_k}.json')
    results_writer = ResultsWriter(results_to_rerank)
    results_writer.write_in_json_format(results_output_path)
    print(f"Results saved to {results_output_path}")

def to_llm_results(results, queries, corpus, top_k):
    retrieved_results_with_text = []
    for qid, docs_scores in results.items():
        query_text = queries[qid]
        for doc_id, score in docs_scores.items():
            doc_text = corpus[doc_id]
            result_with_text = {
                'qid': qid,
                'query_text': query_text,
                'doc_id': doc_id,
                'doc_text': doc_text,
                'score': score
            }
            retrieved_results_with_text.append(result_with_text)

    hits_by_query = defaultdict(list)
    for result in retrieved_results_with_text:
        content_string = ''
        if result['doc_text'].get('title'):
            content_string += result['doc_text']['title'] + ". "
        content_string += result['doc_text']['text']
        hits_by_query[result['query_text']].append({
            'qid': result['qid'],
            'docid': result['doc_id'],
            'score': result['score'],
            'content': content_string
        })

    results_to_rerank = []
    for query_text, hits in hits_by_query.items():
        sorted_hits = sorted(hits, reverse=True, key=lambda x: x['score'])[:top_k]
        result = Result(query=query_text, hits=sorted_hits)
        results_to_rerank.append(result)
    
    return results_to_rerank

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--top_k', required=True, type=int)
    args = parser.parse_args()

    assert args.data_type in ["beir"], "Invalid data_type. Must be 'beir' ."

    convert_results(args.output_dir, args.data_dir, args.dataset, args.data_type, args.top_k)
