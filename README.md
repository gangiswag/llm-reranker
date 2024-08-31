# FIRST: Faster Improved Listwise Reranking with Single Token Decoding
Relevance Feeback code will be released shortly after!


## Installation
You need to install the tevatron library (original source [here](https://github.com/texttron/tevatron)) which provides the framework for retrieval.

```
conda create --name {your env name} python=3.9.18
cd tevatron
pip install --editable .
pip install beir
```
## You need to install the vLLM library (Instruction [here](https://docs.vllm.ai/en/latest/getting_started/installation.html)) which provides optimization for LLM generation.

Before running, do
```
export REPO_DIR=<path to this directory e.g. /shared/nas/data/m1/revanth3/exp/prf/ai2_data/workspace/repo/llm-reranker>
```

## 1. Retrieval
Please download the precomputed BEIR encodings stored at (Link will be added shortly)
Run the baseline Contriever retrieval using the precomputed encodings

```
bash bash/beir/run_1st_retrieval.sh <Path of precomputed BEIR encodings>
```
To get the baseline contriever scores and preprocess datasets, Run:

```
bash bash/beir/run_eval.sh rank
```

## 2. Reranking
### 2a. Baseline Cross-encoder reranking
Cross-encoder rerankig config is at `{REPO_DIR}/bash/beir/run_rerank_CE.sh`
To run the baseline cross encoder re-ranking, run:
```
bash bash/beir/run_rerank.sh
```
### 2b. LLM Reranking
LLM results preparation config is at `{REPO_DIR}/bash/beir/run_convert_results.sh`
To prepare retrieval results for LLM reranking, run:

```
bash bash/beir/run_convert_results.sh
```

LLM rerankig config is at `{REPO_DIR}/bash/beir/run_rerank_llm.sh`
To run the LLM reranking, run:

```
bash bash/beir/run_rerank_llm.sh
```

Evaluation config is at `{REPO_DIR}/bash/beir/run_eval.sh`
To verify that ranking performance has improved from reranking, run:
```
bash bash/run_eval.sh rerank

Set flag --suffix to "llm_FIRST_alpha" for FIRST LLM evaluation or "ce" for cross encoder reranker
```


### 3. Train

We support three training objectives: Ranking, Generation, and Combined. The Ranking objective uses a learning-to-rank algorithm to output the logits for the highest-ranked passage ID. The Generation objective follows the principles of Causal Language Modeling, focusing on permutation generation. The Combined objective, which we introduce in our paper, is a novel weighted approach that seamlessly integrates both ranking and generation principles, and is the setting applied to FIRST model.

Training and accelerate configs are at `{REPO_DIR}/bash/run_train.sh` and `{REPO_DIR}/train_configs/accel_config.yaml`, respectively.

To train the model, run:
```
bash bash/run_train.sh
```

