# FIRST: Faster Improved Listwise Reranking with Single Token Decoding


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
export REPO_DIR=<path to this directory e.g. /shared/nas/data/m1/revanth3/exp/prf/ai2_data/workspace/repo/first_llm>
```

## Retrieval Result

Please download the Contriever Outputs [here](https://drive.google.com/drive/folders/1eMiqwiTVwJy_Zcss7LQF9hQ1aeTFMZUm?usp=sharing) and insert into outputs/beir/

To get the contriever scores:
```
bash bash/beir/run_eval.sh rank
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
