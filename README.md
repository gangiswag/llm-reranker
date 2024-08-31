# FIRST: Faster Improved Listwise Reranking with Single Token Decoding
Relevance Feeback code will be released shortly afterward!


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

## Retrieval
Please download the precomputed BEIR encodings stored at (Link will be added shortly)
Run the baseline Contriever retrieval using the precomputed encodings

```
bash bash/beir/run_1st_retrieval.sh <Location of precomputed BEIR encodings>
```
To get the baseline contriever scores and preprocess datasets:

```
bash bash/beir/run_eval.sh rank
```

## Reranking
To run the baseline cross encoder re-ranking:
```
bash bash/beir/run_rerank.sh
```
### LLM Reranking

To prepare retrieval results for LLM reranking:

```
bash bash/beir/run_convert_results.sh
```

To run the LLM reranking:

```
bash bash/beir/run_rerank_llm.sh
```

To verify that ranking performance has improved from reranking, run:
```
bash bash/run_eval.sh rerank

Set flag --suffix to "llm_FIRST_alpha" for FIRST LLM evaluation or "ce" for cross encoder reranker
```

### Train

We support three training objectives, Ranking, Generation, and Combined. Ranking objective applies learning-to-rank algorithm to output logits of first ranked passage ID, and Generation objective is the same to Causal Langauge Modeling of Permutation Generation. Combined corresponds to the novel weighted objective setting in which FIRST was trained on.

Training and accelerate configs are at `{REPO_DIR}/bash/run_train.sh` and `{REPO_DIR}/train_configs/accel_config.yaml`, respectively.

To train the model, run:
```
bash bash/run_train.sh
```

