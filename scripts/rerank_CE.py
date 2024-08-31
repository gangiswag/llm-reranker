from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import csv
import os
import logging
import json
from argparse import ArgumentParser

def rerank_beir_outputs(output_path, data_dir, dataset, data_type, top_k):
    if data_type == "beir":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(data_dir, "datasets")
        data_path = util.download_and_unzip(url, out_dir)
    else:
        data_path = data_dir + dataset

    if dataset == "msmarco":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    model_output_path = os.path.join(os.path.join(output_path, dataset), "rank.tsv")
    csv_reader = csv.reader(open(model_output_path), delimiter="\t", quotechar='|')
    results = dict()
    for row in csv_reader:
        qid = str(row[0])
        pid = str(row[1])
        score = float(row[2])
        if qid not in results:
            results[qid] = dict()
        results[qid][pid] = score
    
    retriever = EvaluateRetrieval()
        
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    print("Retriever evaluation")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10,20,100])
    print(ndcg)
    mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100], "mrr")
    print(mrr)
    if dataset == "trec-covid":
        recall_cap = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100], "recall_cap")
        print(recall_cap)
    else:
        print(recall)

    if data_type == "beir":
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    else:
        cross_encoder_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    reranker = Rerank(cross_encoder_model, batch_size=64)

    rerank_results = reranker.rerank(corpus, queries, results, top_k=int(top_k))


    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    print("Re-ranker evaluation")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, [1,3,5,10,20,100])
    print(ndcg)
    mrr = retriever.evaluate_custom(qrels, rerank_results, [1,3,5,10,20,100], "mrr")
    print(mrr)
    if dataset == "trec-covid":
        recall_cap = retriever.evaluate_custom(qrels, rerank_results, [1,3,5,10,20,100], "recall_cap")
        print(recall_cap)
    else:
        print(recall)

    rerank_path = os.path.join(os.path.join(output_path, dataset), "rerank_" + str(top_k) + "_ce.json")
    json.dump(rerank_results, open(rerank_path, "w"), indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--top_k', required=True)
    args = parser.parse_args()

    assert args.data_type == "beir" or args.data_type == "mrtydi"

    rerank_beir_outputs(args.output_dir, args.data_dir, args.dataset, args.data_type, args.top_k)
