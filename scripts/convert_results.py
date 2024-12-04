from beir.datasets.data_loader import GenericDataLoader
import csv
import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from utils.result import Result, ResultsWriter

def convert_results(output_path, data_dir, dataset, data_type, top_k, rerank_type="text"):
    """Convert ranking results to format suitable for reranking
    
    Args:
        rerank_type (str): Whether this is for "text" or "code" reranking
    """
    print(f"Loading {dataset} dataset")
    
    try:
        # Load datasets based on type
        if rerank_type == "code":
            if dataset in ('swebench_function', 'swebench_file'):
                return convert_results_swebench(output_path, data_dir, dataset, data_type, top_k)
            
            data_path = os.path.join(data_dir, "code_datasets", dataset)
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            # Load rank data
            rank_path = os.path.join(output_path, "code_datasets", dataset, "rank.tsv")
            dataset_output_path = os.path.join(output_path, "code_datasets", dataset)
            
        else:  # text reranking
            out_dir = os.path.join(data_dir, "beir")
            data_path = os.path.join(out_dir, dataset)
            
            split = "dev" if dataset == "msmarco" else "test"
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            
            # Load rank data
            dataset_output_path = os.path.join(output_path, "beir", dataset)
            rank_path = os.path.join(dataset_output_path, "rank.tsv")

        print("Loading rank data")
        if not os.path.exists(rank_path):
            print(f"Rank file not found: {rank_path}")
            return

        results = {}
        with open(rank_path, 'r') as rank_file:
            csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
            if rerank_type == "code":
                next(csv_reader)  # Skip header for code files
            for row in csv_reader:
                qid = str(row[0])
                pid = str(row[1])
                score = float(row[2])
                if qid not in results:
                    results[qid] = {}
                results[qid][pid] = score

        print("Converting to reranker results")
        # Remove dummy entries if present (for code reranking)
        if 'dummy' in results:
            results.pop('dummy')
            
        results_to_rerank = to_reranker_results(results, queries, corpus, top_k)

        # Ensure output directory exists
        os.makedirs(dataset_output_path, exist_ok=True)
        
        results_output_path = os.path.join(dataset_output_path, f'rank_{top_k}.json')
        results_writer = ResultsWriter(results_to_rerank)
        results_writer.write_in_json_format(results_output_path)
        print(f"Results saved to {results_output_path}")
        
    except Exception as e:
        print(f"Error in convert_results: {e}")
        raise

def convert_results_swebench(output_path, data_dir, dataset, data_type, top_k):
    """Special handling for swebench datasets"""
    prefx = f"csn_{dataset.split('_')[1]}"
    instance_list = [instance for instance in os.listdir(data_dir) if instance.startswith(prefx)]
    
    for dataset_instance in instance_list:
        print(f"Loading {dataset_instance} dataset")
        data_path = os.path.join(data_dir, "code_datasets", dataset_instance)

        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

            print("Loading rank data")
            rank_path = os.path.join(output_path, "code_datasets", dataset_instance, "rank.tsv")
            
            results = {}
            with open(rank_path, 'r') as rank_file:
                csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    qid = str(row[0])
                    pid = str(row[1])
                    score = float(row[2])
                    if qid not in results:
                        results[qid] = {}
                    results[qid][pid] = score

            print("Converting to reranker results")
            results_to_rerank = to_reranker_results(results, queries, corpus, top_k)

            dataset_output_path = os.path.join(output_path, "code_datasets", dataset_instance)
            os.makedirs(dataset_output_path, exist_ok=True)
            
            results_output_path = os.path.join(dataset_output_path, f'rank_{top_k}.json')
            results_writer = ResultsWriter(results_to_rerank)
            results_writer.write_in_json_format(results_output_path)
            print(f"Results saved to {results_output_path}")
            
        except Exception as e:
            print(f"Error processing {dataset_instance}: {e}")
            continue

def to_reranker_results(results, queries, corpus, top_k):
    """Convert results to format needed by reranker"""
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
        if isinstance(result['doc_text'], dict):
            if result['doc_text'].get('title'):
                content_string += result['doc_text']['title'] + ". "
            content_string += result['doc_text']['text']
        else:
            content_string = result['doc_text']
            
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
    parser.add_argument('--rerank_type', type=str, default="text", choices=["text", "code"],
                      help="Whether to convert for code or text reranking")
    args = parser.parse_args()

    assert args.data_type in ["beir", "codedataset"], "Invalid data_type. Must be 'beir' or 'codedataset'."
    
    if args.data_type == "codedataset" and args.rerank_type != "code":
        print("Warning: codedataset data_type implies code reranking. Setting rerank_type to 'code'")
        args.rerank_type = "code"

    convert_results(args.output_dir, args.data_dir, args.dataset, args.data_type, args.top_k, args.rerank_type)
