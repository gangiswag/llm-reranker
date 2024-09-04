from copy import deepcopy
import csv
from itertools import count
import json
import sys
csv.field_size_limit(sys.maxsize)

from tqdm import tqdm
from collections import Counter
import random
import pickle
import glob
from argparse import ArgumentParser
import os
import torch
import pickle
import numpy as np
import torch.nn as nn


def get_all_examples(rank_path, psg_embs_dir, qry_embs_path, output_path):
    import os
    import torch
    import pickle
    import numpy as np
    import torch.nn as nn

    embs_path = os.path.join(psg_embs_dir, "split*.pt")
    embs_files = glob.glob(embs_path)
    embs = list()
    ids = list()
    p_reps_0, look_up_0 = pickle.load(open(embs_files[0], "rb"))
    embs.extend(p_reps_0)
    ids.extend(look_up_0)
    for f in tqdm(embs_files[1:]):
        p_reps, look_up = pickle.load(open(f, "rb"))
        embs.extend(p_reps)
        ids.extend(look_up)   
    embs = np.array(embs, dtype=np.float32)
    embs_dict = dict()
    assert len(embs) == len(ids)
    for emb, pid in zip(embs, ids):
        embs_dict[pid] = emb

    q_reps, q_look_up = pickle.load(open(qry_embs_path, "rb"))
    q_embs = np.array(q_reps, dtype=np.float32)
    assert len(q_embs) == len(q_look_up)
    q_tuples = list()
    for q_emb, qid in zip(q_embs, q_look_up):
        q_tuples.append((str(qid), q_emb))

    results = dict()
    csv_reader = csv.reader(open(rank_path), delimiter="\t", quotechar='|')    
    for row in csv_reader:
        qid = str(row[0])
        pid = str(row[1])
        score = float(row[2])
        if qid not in results:
            results[qid] = dict()
        results[qid][pid] = score

    examples = list()
    for qid, q_emb in tqdm(q_tuples):
        item = dict() 
        passage_ids = sorted(results[qid].items(), key=lambda item: item[1], reverse=True)
        passage_ids = [i[0] for i in passage_ids]
        
        item["query_rep"] = q_emb
        item["passage_ids"] = deepcopy(passage_ids)
        item["passage_embs"] = list()
        for pid in item["passage_ids"]:
            item["passage_embs"].append(embs_dict[pid])
        item["passage_embs"] = np.array(item["passage_embs"])
        item["query_id"] = qid
        item["qrels"] = ""
        examples.append(deepcopy(item))
    pickle.dump(examples, open(output_path, "wb"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rank_path', required=True)
    parser.add_argument('--psg_embs_dir', required=True)
    parser.add_argument('--qry_embs_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    get_all_examples(args.rank_path, args.psg_embs_dir, args.qry_embs_path, args.output_path)


