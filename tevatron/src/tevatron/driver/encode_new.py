import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import json

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
import csv

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.modeling import EncoderOutput, DenseModel
from tevatron.utils.normalize_text import normalize
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    corpus = list()

    if data_args.encode_is_qry:
        tsv_reader = csv.reader(open(data_args.encode_in_path[0]), delimiter="\t")
        for row in tsv_reader:
            corpus.append({"id": row[0], "text": row[1]})
        corpus_items = [(c["id"],c["text"]) for c in corpus]
    
    else:
        corpus_lines = open(data_args.encode_in_path[0]).readlines()
        for line in corpus_lines:
            corpus.append(deepcopy(json.loads(line)))

        corpus_items = [(c["id"], c["title"] + " " +c["text"]) for c in corpus]

    if data_args.normalize_text:
        corpus_items = [(c[0], normalize(c[1])) for c in corpus_items]
    
    if data_args.lower_case:
        corpus_items = [(c[0], c[1].lower()) for c in corpus_items]

    corpus_items = sorted(corpus_items, key=lambda item: len(item[1]), reverse=True)

    batch_size = training_args.per_device_eval_batch_size 
    nbatch = int(len(corpus_items) / batch_size) + 1

    with torch.no_grad():
        for k in tqdm(range(nbatch)):
            try:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                batch_items = corpus_items[start_idx:end_idx]
                batch_ids = [item[0] for item in batch_items]
                batch_text = [item[1] for item in batch_items]
                
                cencode = tokenizer(
                        batch_text,
                        padding=True,
                        truncation='longest_first',
                        return_tensors="pt",
                    )

                cencode = {key: value.cuda() for key, value in cencode.items()}

                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=cencode)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=cencode)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())
                
                lookup_indices.extend(batch_ids)
            except Exception as e:
                print(e)
                continue


    encoded = np.concatenate(encoded)

    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)




if __name__ == "__main__":
    main()