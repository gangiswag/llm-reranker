from datetime import datetime
from pathlib import Path
from typing import List
import os
import time

from tqdm import tqdm

from utils.rankllm import RankLLM
from utils.result import Result

class Reranker:
    def __init__(self, agent: RankLLM) -> None:
        self._agent = agent

    def rerank(
        self,
        retrieved_results: List[Result],
        use_logits: bool = False,
        use_alpha: bool = False,
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        logging: bool = False,
        batched: bool = False
    ) -> List[Result]:
        """
        Reranks a list of retrieved results using the RankLLM agent.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM agent to obtain a new ranking.

        Args:
            retrieved_results (List[Result]): The list of results to be reranked.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked results.
        """
        if batched:
            return self._agent.sliding_windows_batched(
                retrieved_results,
                use_logits=use_logits,
                use_alpha=use_alpha,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(retrieved_results[0].hits)), #TODO: Fails arbitrary hit sizes
                window_size=window_size,
                step=step,
                logging=logging,
            )

        rerank_results = []
        for result in tqdm(retrieved_results):
            rerank_result = self._agent.sliding_windows(
                result,
                use_logits=use_logits,
                use_alpha=use_alpha,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(result.hits)),
                window_size=window_size,
                step=step,
                logging=logging,
            )
            rerank_results.append(rerank_result)
        return rerank_results
