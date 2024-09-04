import os
import json
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from utils.loss import loss_dict

class QueryImpModel(nn.Module):
    def __init__(self, query_rep, scaler):
        super().__init__()
        self.query_rep = nn.Parameter(torch.FloatTensor(query_rep), requires_grad=True)
        self.scaler = scaler

    def forward(self, psg_embs: Tensor, attn_mask: Tensor = None):
        pred_scores = (self.scaler / 2) * torch.matmul(self.query_rep, psg_embs.transpose(0, 1))
        if attn_mask is not None:
            extended_attention_mask = (1.0 - attn_mask) * torch.finfo(pred_scores.dtype).min
            pred_scores += extended_attention_mask
        pred_probs = nn.functional.log_softmax(pred_scores, dim=-1)
        return pred_probs


class QueryScoreModel(nn.Module):
    def __init__(self, query_rep, scaler=2.0):
        super().__init__()
        self.query_rep = nn.Parameter(torch.FloatTensor(query_rep), requires_grad=True)
        self.scaler = scaler

    def forward(self, psg_embs: Tensor, attn_mask: Tensor = None):
        pred_scores = (self.scaler / 2) * torch.matmul(self.query_rep, psg_embs.transpose(0, 1))
        if attn_mask is not None:
            extended_attention_mask = (1.0 - attn_mask) * torch.finfo(pred_scores.dtype).min
            pred_scores += extended_attention_mask
        return pred_scores.unsqueeze(0)


def load_results(inp_path, rerank_path, ce_top_k, llm_top_k, use_logits, use_alpha):
    llm_rerank = None
    ce_rerank = None

    if llm_top_k > 0:
        suffix = "_llm"
        suffix += "_FIRST" if use_logits else "_gen"
        suffix += "_alpha" if use_alpha else "_num"
        llm_rerank = json.load(open(os.path.join(rerank_path, f"rerank_{llm_top_k}{suffix}.json")))

    if ce_top_k > 0:
        ce_rerank = json.load(open(os.path.join(rerank_path, f"rerank_{ce_top_k}_ce.json")))

    examples = pickle.load(open(inp_path, "rb"))
    return examples, ce_rerank, llm_rerank


def prepare_distill_ce(data, ce_rerank, ce_top_k):
    qid = data["query_id"]
    pids = data["passage_ids"][:ce_top_k]

    data_passage_mapping = {pid: deepcopy(emb) for pid, emb in zip(data["passage_ids"], data["passage_embs"])}
    target_scores = [ce_rerank[qid][pid] for pid in pids]
    psg_embs = [data_passage_mapping[pid] for pid in pids]

    target_scores = torch.FloatTensor(target_scores)
    target_probs = nn.functional.log_softmax(target_scores, dim=-1)

    baseline_rep = torch.FloatTensor(data["query_rep"])
    passage_reps = torch.FloatTensor(np.array(psg_embs))

    init_scores = torch.matmul(baseline_rep, passage_reps.transpose(0, 1))
    scaler = (target_scores.max() - target_scores.min()) / (init_scores.max().item() - init_scores.min().item())

    return passage_reps, target_probs, scaler


def prepare_distill_llm(data, llm_rerank, query_rep, llm_top_k):
    qid = data["query_id"]
    pids = data["passage_ids"][:llm_top_k]

    data_passage_mapping = {pid: deepcopy(emb) for pid, emb in zip(data["passage_ids"], data["passage_embs"])}
    reranked_target_scores = [llm_rerank[qid][pid] for pid in pids]
    reranked_psg_embs = [data_passage_mapping[pid] for pid in pids]

    reranked_target_scores = torch.FloatTensor(reranked_target_scores)
    reranked_passage_reps = torch.FloatTensor(np.array(reranked_psg_embs))

    init_scores = torch.matmul(query_rep, reranked_passage_reps.transpose(0, 1))
    scaler = (reranked_target_scores.max() - reranked_target_scores.min()) / \
             (init_scores.max().item() - init_scores.min().item())

    return reranked_passage_reps, reranked_target_scores.unsqueeze(0), scaler


def run_query_teacher_importance_learner(inp_path, rerank_path, output_path, loss_path, ce_top_k, llm_top_k, learning_rate,
                                         num_updates, use_logits, use_alpha, llm_loss):
    assert llm_loss in loss_dict
    examples, ce_rerank, llm_rerank = load_results(inp_path, rerank_path, ce_top_k, llm_top_k, use_logits, use_alpha)

    reps = []
    ids = []

    for data in tqdm(examples):
        baseline_rep = torch.FloatTensor(data["query_rep"])

        try:
            learned_rep = baseline_rep
            if ce_top_k > 0:
                passage_reps, target_probs, scaler = prepare_distill_ce(data, ce_rerank, ce_top_k)
                ce_dstl_model = QueryImpModel(query_rep=baseline_rep.numpy(), scaler=scaler)
                loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True)
                optimizer = optim.Adam(ce_dstl_model.parameters(), lr=learning_rate)

                for _ in range(num_updates):
                    optimizer.zero_grad()
                    pred_probs = ce_dstl_model(psg_embs=passage_reps)
                    loss = loss_function(pred_probs.unsqueeze(0), target_probs.unsqueeze(0))
                    loss.backward()
                    optimizer.step()

                learned_rep = ce_dstl_model.query_rep.data.cpu().detach()

            reranked_passage_reps, reranked_target_scores, scaler = prepare_distill_llm(data, llm_rerank, learned_rep,
                                                                                        llm_top_k)
            llm_dstl_model = QueryScoreModel(query_rep=learned_rep.numpy(), scaler=scaler)
            optimizer = optim.Adam(llm_dstl_model.parameters(), lr=learning_rate / 5)

            for _ in range(num_updates // 5):
                optimizer.zero_grad()
                pred_scores = llm_dstl_model(psg_embs=reranked_passage_reps)
                loss = loss_dict[llm_loss](pred_scores, reranked_target_scores, weighted=True if llm_loss == "ranknet" else False)
                loss.backward()
                optimizer.step()

            rep = llm_dstl_model.query_rep.data.cpu().detach()
            reps.append(rep.numpy())
            ids.append(data["query_id"])
        except Exception as e:
            print(f"Error for query ID {data['query_id']}: {e}")

    pickle.dump((np.array(reps), ids), open(output_path, "wb"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--inp_path', required=True)
    parser.add_argument('--rerank_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--loss_path', required=True)
    parser.add_argument('--ce_top_k', type=int, default=100)
    parser.add_argument('--llm_top_k', type=int, default=9)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--num_updates', type=int, default=100)
    parser.add_argument('--use_logits', type=int, default=0)
    parser.add_argument('--use_alpha', type=int, default=0)
    parser.add_argument('--llm_loss', type=str, default="lambdarank")

    args = parser.parse_args()

    run_query_teacher_importance_learner(args.inp_path, args.rerank_path, args.output_path, args.loss_path, args.ce_top_k, args.llm_top_k, args.learning_rate, args.num_updates, args.use_logits, args.use_alpha, args.llm_loss)
