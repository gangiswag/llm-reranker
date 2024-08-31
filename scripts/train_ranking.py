import os
import torch
import time
import logging
import json
import math
import argparse
from tqdm import tqdm
from functools import partial
import random
import numpy as np
import bitsandbytes as bnb

import transformers
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
import datasets
from datasets import load_dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, get_scheduler, CONFIG_MAPPING
from torch.utils.data import DataLoader, Dataset

from rerank_llm import evaluate_results
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

from utils.loss import lambdarank, listNet, rank_net
from utils.dataset import RankingDataset, GenerationDataset, ranking_collate_fn, generation_collate_fn, combined_collate_fn
from utils.train_utils import load_data, NEFTune, parse_args

logger = get_logger(__name__)

max_psg_num = 20
START_IDX = ord('A')

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(cpu=False, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], mixed_precision="bf16", gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name, cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.torch.bfloat16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, cache_dir=args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})

    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
    )

    # TODO: Load dataset file. (Skip eval for now)
    if not os.path.isfile(args.train_dataset_path):
        # Using Hugging Face dataset
        ds = load_dataset(args.train_dataset_path)
        raw_train_data = ds['train']
    else:
        raw_train_data = load_data(args.train_dataset_path)

    if not os.path.isfile(args.eval_dataset_path):
        ds = load_dataset(args.eval_dataset_path)
        raw_eval_data = ds['test']
    else:
        raw_eval_data = load_data(args.eval_dataset_path)

    if args.objective == "generation":
        train_dataset = GenerationDataset(raw_train_data, tokenizer)
        train_collate_fn = generation_collate_fn
    elif args.objective == "combined":
        train_dataset = GenerationDataset(raw_train_data, tokenizer, combined=True)
        train_collate_fn = combined_collate_fn
    else:
        train_dataset = RankingDataset(raw_train_data, tokenizer, type="train")        
        train_collate_fn = ranking_collate_fn

    eval_dataset = RankingDataset(raw_eval_data, tokenizer, type="eval")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(train_collate_fn, tokenizer=tokenizer), batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=partial(ranking_collate_fn, tokenizer=tokenizer), batch_size=args.per_device_eval_batch_size)
    
    # Load MSMARCO for evaluation
    out_dir = os.path.join(args.beir_data_path, "datasets")
    data_path = os.path.join(out_dir, "msmarco")
    qrels = []
    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="dev") # Commented out to test generation training

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Add noise to embedding if using NEFTune
    if args.noisy_embedding_alpha is not None:
        model = NEFTune(model, args.noisy_embedding_alpha)

    # Prepare everything with our `accelerator`.
    train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, lr_scheduler
    )

    device = accelerator.device

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("train_ranking", experiment_config)

    if args.objective != "generation":
        if args.ranking_loss == "lambda":
            ranking_loss_fn = lambdarank
        elif args.ranking_loss == "listnet":
            ranking_loss_fn = listNet
        elif args.ranking_loss == "ranknet":
            ranking_loss_fn = rank_net
        else:
            raise Exception("Invalid ranking loss specified. Choose from [lambda, listnet, ranknet]")
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
        
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    model.train()

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.with_tracking:
            step_loss = 0
            if args.objective == "combined":
                step_rank_loss = 0
                step_generate_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.autocast():
                with accelerator.accumulate(model):
                    if args.objective == "generation" or args.objective == "combined":
                        if args.objective == "generation":
                            tokenized_input, label = batch
                        else:
                            tokenized_input, label, rank_labels, source_lens = batch
                        tokenized_input = tokenized_input.to(device)
                    
                        outputs = model(tokenized_input)
                        logits = outputs.logits
                        logits = logits[..., :-1, :].contiguous()

                        # shift target for causal langauge modeling
                        labels = label[..., 1:].contiguous()
                        log_probs = -nn.functional.log_softmax(logits, dim=-1)
                        if labels.dim() == log_probs.dim() - 1:
                            labels = labels.unsqueeze(-1)

                        epsilon = 0.0
                        ignore_index = -100
                        padding_mask = labels.eq(ignore_index)

                        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
                        # will ignore them in any case.
                        labels = torch.clamp(labels, min=0)
                        nll_loss = log_probs.gather(dim=-1, index=labels)
                        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

                        nll_loss.masked_fill_(padding_mask, 0.0)
                        smoothed_loss.masked_fill_(padding_mask, 0.0)

                        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
                        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
                        nll_loss = nll_loss.sum() / num_active_elements
                        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
                        loss =  (1 - epsilon) * nll_loss + epsilon * smoothed_loss       
                        generate_loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss      

                        if args.objective == "combined":
                            rank_losses = torch.zeros(len(rank_labels))
                            for batch_index, true_rank in enumerate(rank_labels):
                                pred_logits = outputs.logits[batch_index]
                                rank_pred_index = source_lens[batch_index]
                                pred = pred_logits[rank_pred_index]
                                gather_indices = [tokenizer.convert_tokens_to_ids(chr(c)) for c in range(START_IDX, START_IDX+len(true_rank))]
                                scores = torch.gather(pred, 0, torch.tensor(gather_indices).to(device))
                                scores_sorted, _ = scores.sort(descending=True, dim=-1)
                                true_scores = torch.gather(scores_sorted, 0, torch.tensor(true_rank).to(device))
                                if args.ranking_loss == "ranknet":
                                    rank_losses[batch_index] = ranking_loss_fn(scores.unsqueeze(0), true_scores.unsqueeze(0), weighted=args.weighted) 
                                else:
                                    rank_losses[batch_index] = ranking_loss_fn(scores.unsqueeze(0), true_scores.unsqueeze(0)) 
                            rank_loss = rank_losses.mean().to(device)
                            if not torch.isnan(rank_loss):
                                if args.ranking_loss == "listnet":
                                    loss += torch.mul(rank_loss, 0.1)
                                elif args.ranking_loss == "ranknet" and args.weighted:
                                    loss += torch.mul(rank_loss, 10.0)
                                else:
                                    loss += rank_loss                                    
                                
                    else:
                        tokenized_input, label = batch
                        tokenized_input = tokenized_input.to(device)

                        outputs = model(**tokenized_input)
                        pred = outputs.logits[:,-1]

                        losses = torch.zeros(len(label))
                        for batch_index, item in enumerate(label):
                            gather_indices = [tokenizer.convert_tokens_to_ids(chr(c)) for c in range(START_IDX, START_IDX+len(item))]
                            scores = torch.gather(pred[batch_index], 0, torch.tensor(gather_indices).to(device))
                            scores_sorted, _ = scores.sort(descending=True, dim=-1)
                            true_scores = torch.gather(scores_sorted, 0, torch.tensor(item).to(device))
                            losses[batch_index] = ranking_loss_fn(scores.unsqueeze(0), true_scores.unsqueeze(0))

                        loss = losses.mean().to(device)

                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        step_loss += loss.detach().float()
                        if args.objective == "combined":
                            step_rank_loss += rank_loss.detach().float()
                            step_generate_loss += generate_loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1
                if args.with_tracking:
                    accelerator.log(
                        {
                            # "perplexity": perplexity,
                            # "eval_loss": eval_loss,
                            "train_loss": step_loss,
                        },
                        step=completed_steps,
                    )
                    if args.objective == "combined":
                        accelerator.log(
                        {
                            # "perplexity": perplexity,
                            "generate_loss": step_generate_loss,
                            "rank_loss": step_rank_loss,
                        },
                        step=completed_steps,
                        )
                        step_rank_loss = 0
                        step_generate_loss = 0
                    step_loss = 0
                progress_bar.update(1)
                
                # Evaluation
                if completed_steps > 0 and completed_steps % args.eval_steps == 0:                 
                    model.eval()
                    eval_results = dict()

                    for batch in tqdm(eval_dataloader):
                        tokenized_input, label = batch
                        tokenized_input = tokenized_input.to(device)
                        
                        with torch.no_grad():
                            outputs = model(**tokenized_input)
                            pred = outputs.logits[:,-1]

                        for batch_index, item in enumerate(label):
                            qid = item[0]
                            docids = item[1:101]
                            doc_scores = item[101:]

                            eval_results[qid] = dict()

                            gather_indices = [tokenizer.convert_tokens_to_ids(chr(c)) for c in range(START_IDX, START_IDX+max_psg_num)]
                            scores = torch.gather(pred[batch_index], 0, torch.tensor(gather_indices).to(device))
                            sort_idx = list(scores.argsort(descending=True, dim=-1).detach().cpu().numpy()) + [i for i in range(20,100)]
                            sorted_docids = np.array(docids)[sort_idx]

                            for idx, docid in enumerate(sorted_docids):
                                eval_results[qid][docid] = doc_scores[idx]

                    current_level = logging.getLogger().getEffectiveLevel()
                    logging.getLogger().setLevel(logging.WARNING)
                    ndcg, _map, recall, precision = evaluate_results("msmarco", qrels, eval_results)
                    logging.getLogger().setLevel(current_level)
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "eval_ndcg": ndcg["NDCG@10"],
                            },
                            step=completed_steps,
                        )
                    # Back to training
                    model.train()

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    # accelerator.save_state(output_dir)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
                    )  
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            # accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
            )  
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()