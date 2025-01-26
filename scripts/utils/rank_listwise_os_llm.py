import os
import json
import random
from typing import Optional, Tuple, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import numpy as np
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from transformers.generation import GenerationConfig
from vllm import LLM, SamplingParams, RequestOutput

from utils.rankllm import PromptMode, RankLLM
from utils.result import Result

ALPH_START_IDX = ord('A') - 1

class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        batched: bool = False,
        rerank_type: str = "text",
        code_prompt_type: str = "docstring",
    ) -> None:
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available on this device"
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. Only RANK_GPT is supported."
            )

        self._llm = LLM(model=model, max_logprobs=30, enforce_eager=True, gpu_memory_utilization=0.95, max_model_len=32768, trust_remote_code=True, enable_chunked_prefill=False, tensor_parallel_size=1)
        self._tokenizer = self._llm.get_tokenizer()
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self._batched = batched
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        self._rerank_type = rerank_type
        self._code_prompt_type = code_prompt_type

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]

    def _evaluate_logits(self, logits: Dict[str, 'Logit'], use_alpha: bool, total: Tuple[int, int]) -> Tuple[str, Dict[int, float]]:
        if use_alpha:
            evaluations = {
                ord(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if len(logit.decoded_token) == 1 and 
                   logit.decoded_token.isalpha() and 
                   ALPH_START_IDX + 1 <= ord(logit.decoded_token) <= ALPH_START_IDX + self._window_size
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{chr(x)}]" for x, y in sorted_evaluations])
        else:
            evaluations = {
                int(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if logit.decoded_token.isnumeric() and
                   not unicodedata.name(logit.decoded_token).startswith(('SUPERSCRIPT', 'VULGAR FRACTION', 'SUBSCRIPT')) and
                   total[0] <= int(logit.decoded_token) <= total[1]
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{x}]" for x, y in sorted_evaluations])

        return result_string, evaluations

    def _get_logits_single_digit(self, output: RequestOutput, use_alpha: bool = False, effective_location: int = 1, total: Tuple[int, int] = (1, 9)):
        logits = output.outputs[0].logprobs[effective_location]
        return self._evaluate_logits(logits, use_alpha, total)

    def _get_logits_single_digit_batched(self, output: RequestOutput, use_alpha: bool = False, effective_location: int = 1, total: Tuple[int, int] = (1, 9)):
        """Batched version of _get_logits_single_digit"""
        logits = output.outputs[0].logprobs[effective_location]
        return self._evaluate_logits(logits, use_alpha, total)

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None, use_logits: bool = False, use_alpha: bool = False
    ) -> Tuple[str, int]:
        """Run the language model with appropriate restrictions for code vs text reranking"""
        if self._rerank_type == "code" and (use_logits or use_alpha):
            print("Warning: Code reranking does not support logits or alpha mode. Defaulting to standard mode.")
            use_logits = False
            use_alpha = False

        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size
        if use_logits:
            if use_alpha:
                max_new_tokens = 1
                min_new_tokens = 1
                params = SamplingParams(min_tokens=min_new_tokens, max_tokens=max_new_tokens, temperature=temp, logprobs=120)
                output = self._llm.generate([prompt+"["], sampling_params=params, use_tqdm=False)
                output = output[0]

                s, evaluatearr = self._get_logits_single_digit(output, effective_location=0, use_alpha=use_alpha)
            else:
                max_new_tokens = 1
                min_new_tokens = 1
                params = SamplingParams(min_tokens=min_new_tokens, max_tokens=max_new_tokens, temperature=temp, logprobs=100)
                output = self._llm.generate([prompt+"["], sampling_params=params, use_tqdm=False)
                output = output[0]

                s, evaluatearr = self._get_logits_single_digit(output, effective_location=0)

            return s, len(s)
        else:
            max_new_tokens = self.num_output_tokens(use_alpha, current_window_size)
            min_new_tokens = self.num_output_tokens(use_alpha, current_window_size)
            params = SamplingParams(min_tokens=min_new_tokens, max_tokens=max_new_tokens, temperature=temp)
            output = self._llm.generate([prompt], sampling_params=params, use_tqdm=False)[0]
            return output.outputs[0].text, len(output.outputs[0].text)

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
        use_logits: bool = False,
        use_alpha: bool = False,
    ) -> List[Tuple[str, int]]:
        """Run batched inference with appropriate restrictions for code vs text reranking"""
        if self._rerank_type == "code" and (use_logits or use_alpha):
            print("Warning: Code reranking does not support logits or alpha mode. Defaulting to standard mode.")
            use_logits = False
            use_alpha = False

        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size
        if use_logits:
            max_new_tokens = 2
            min_new_tokens = 2
            if use_alpha:
                params = SamplingParams(
                    min_tokens=min_new_tokens,
                    max_tokens=max_new_tokens, 
                    temperature=temp,
                    logprobs=30,
                )
            else:
                assert current_window_size <= 9, "using logits with numerical ordering can only supports window size <= 9"
                params = SamplingParams(
                    min_tokens=min_new_tokens, 
                    max_tokens=max_new_tokens, 
                    temperature=temp,
                    logprobs=30,
                )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            arr = [self._get_logits_single_digit_batched(output, use_alpha=use_alpha) for output in outputs]
            return [(s, len(s)) for s, __ in arr]
        else:
            params = SamplingParams(
                temperature=temp,
                max_tokens=self.num_output_tokens(use_alpha, current_window_size),
                min_tokens=self.num_output_tokens(use_alpha, current_window_size),
            )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            return [
                (output.outputs[0].text, len(output.outputs[0].token_ids))
                for output in outputs
            ]

    def num_output_tokens(self, use_alpha: bool, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        if use_alpha:
            token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        else:
            token_str = " > ".join([f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)])

        _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def _add_prefix_prompt(self, use_alpha, query: str, num: int) -> str:
        if self._rerank_type == "code":
            if self._code_prompt_type == "docstring":
                return self._add_prefix_prompt_doc_string(use_alpha, query, num)
            elif self._code_prompt_type == "github_issue":
                return self._add_prefix_prompt_github_issue(use_alpha, query, num)
            else:
                raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")
        else:  # text reranking
            if use_alpha:
                return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
            else:
                return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, use_alpha, query: str, num: int) -> str:
        if self._rerank_type == "code":
            if self._code_prompt_type == "docstring":
                return self._add_post_prompt_doc_string(use_alpha, query, num)
            elif self._code_prompt_type == "github_issue":
                return self._add_post_prompt_github_issue(use_alpha, query, num)
            else:
                raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")
        else:  # text reranking
            if use_alpha:
                example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
            else:
                example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
            return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

    def _add_prefix_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            return f"I will provide you with {num} code functions, each indicated by an alphabetical identifier []. Rank the code functions based on their relevance to the functionality described by the following doc string: {query}.\n"
        else:
            return f"I will provide you with {num} code snippets, each indicated by a numerical identifier []. Rank the code snippets based on their relevance to the functionality described by the following doc string: {query}.\n"

    def _add_post_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Doc String: {query}.\nRank the {num} code snippets above based on their relevance to the functionality described by the doc string. All the code snippets should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

    def _add_prefix_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            return f"I will provide you with {num} code functions, each indicated by a alphabetical identifier []. Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
        else:
            return f"I will provide you with {num} code functions, each indicated by a numerical identifier []. Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"

    def _add_post_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"GitHub Issue: {query}. \nRank the {num} code functions above based on their relevance to contain the faults causing the GitHub issue. All the code functions should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv
    
    def _add_few_shot_examples_messages(self, messages):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        return messages

    def create_prompt(self, result: Result, use_alpha: bool, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query
        num = len(result.hits[rank_start:rank_end])
        max_length = 1024 if self._rerank_type == "code" else 300
        while True:
            messages = list()
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})
            messages = self._add_few_shot_examples_messages(messages)
            prefix = self._add_prefix_prompt(use_alpha, query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                if self._rerank_type == "code":
                    content = hit["content"]
                    content = content.replace("Title: Content: ", "")
                    tokenized_content = self._tokenizer.tokenize(content)
                    content_tokens = tokenized_content[:int(max_length)]
                    truncated_content = self._tokenizer.convert_tokens_to_string(content_tokens)
                    identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                    input_context += f"[{identifier}] {self._replace_number(truncated_content, use_alpha)}\n"
                else:
                    content = hit["content"].replace("Title: Content: ", "").strip()
                    content = " ".join(content.split()[:max_length])
                    identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                    input_context += f"[{identifier}] {self._replace_number(content, use_alpha)}\n"
            input_context += self._add_post_prompt(use_alpha, query, num)
            messages.append({"role": "user", "content": input_context})
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(rank_end - rank_start):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens - self.max_tokens() + self.num_output_tokens(rank_end - rank_start)
                    ) // ((rank_end - rank_start) * 4),
                )
        return prompt, num_tokens

    def create_prompt_batched(
        self,
        results: List[Result],
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, use_alpha, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
