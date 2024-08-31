import numpy as np
import faiss
from tqdm.autonotebook import trange
import time

import logging

logger = logging.getLogger(__name__)

class BaseFaissHNSWRetriever:
    def __init__(self, init_reps: np.ndarray, hnsw_store_n: int = 512, hnsw_ef_search: int = 128, hnsw_ef_construction: int = 64, similarity_metric=faiss.METRIC_INNER_PRODUCT):
        self.index = faiss.IndexHNSWFlat(init_reps.shape[1], hnsw_store_n, similarity_metric)
        self.index.hnsw.efSearch = hnsw_ef_search
        self.index.hnsw.efConstruction = hnsw_ef_construction
        # self.index = faiss.index_factory(init_reps.shape[1], "HNSW" + str(hnsw_store_n))

    def load(self, fname: str):
        self.index = faiss.read_index(fname)

    def save(self, fname: str):
        faiss.write_index(self.index, fname)
    
    def build(self, p_reps: np.ndarray, buffer_size: int = 1000):
        # sq_norms = (p_reps ** 2).sum(1)
        # max_sq_norm = float(sq_norms.max())
        # aux_dims = np.sqrt(max_sq_norm - sq_norms)
        # p_reps = np.hstack((p_reps, aux_dims.reshape(-1, 1)))
        for start in trange(0, p_reps.shape[0], buffer_size):
            self.index.add(p_reps[start : start + buffer_size])
    
    def search(self, q_reps: np.ndarray, k: int):
        # q_reps = np.hstack((q_reps, np.zeros((q_reps.shape[0], 1), dtype=np.float32)))
        return self.index.search(q_reps, k)
    
    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        # q_reps = np.hstack((q_reps, np.zeros((q_reps.shape[0], 1), dtype=np.float32)))
        all_scores = []
        all_indices = []
        for start_idx in trange(0, num_query, batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices

class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in trange(0, num_query, batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
