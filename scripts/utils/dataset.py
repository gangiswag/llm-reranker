import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from ftfy import fix_text

max_psg_num = 20
START_IDX = ord('A')
IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels, sources_tokenized["input_ids_lens"]

class RankingDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, type) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.tokenizer.padding_side="left"
        self.type = type
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        target_generation = conversation[2]["value"]

        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "["
        prompt = fix_text(prompt)

        if self.type == "train":
            label_map = {}
            label_rank = 0
            for token in target_generation:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]

        elif self.type == "eval":
            label = [self.raw_data[index]["id"]] + self.raw_data[index]["docids"] + self.raw_data[index]["scores"]
        else:
            raise Exception("Invalid run type specified for Dataset. Choose from ['train', 'eval']")
        return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

class GenerationDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, combined=False) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        label = conversation[2]["value"]
        label += self.tokenizer.eos_token
        
        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        if self.combined:
            label_map = {}
            label_rank = 0
            for token in conversation[2]["value"]:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            rank_label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]
            return prompt, label, rank_label
        else:
            return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

def ranking_collate_fn(data, tokenizer):
    prompts, labels = list(zip(*data))
    tokenized_inputs = tokenizer(prompts, padding="longest", truncation=False, return_tensors="pt")
    return tokenized_inputs, labels

def generation_collate_fn(data, tokenizer):
    prompts, labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels

def combined_collate_fn(data, tokenizer):
    prompts, labels, rank_labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels, rank_labels, source_lens
