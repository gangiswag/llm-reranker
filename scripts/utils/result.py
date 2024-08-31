import json
from typing import Any, Dict, List


class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)


class Result:
    def __init__(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        ranking_exec_summary: List[RankingExecInfo] = None,
    ):
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)


class ResultsWriter:
    def __init__(self, results: List[Result], append: bool = False):
        self._results = results
        self._append = append

    def write_in_json_format(self, filename: str):
        results = []
        for result in self._results:
            results.append({"query": result.query, "hits": result.hits})
        with open(filename, "a" if self._append else "w") as f:
            json.dump(results, f, indent=2)

class ResultsLoader:
    def __init__(self, filename: str):
        data = json.load(open(filename, 'r'))
        self._results = []
        for item in data:
            hits = []
            for hit in item['hits']:
                hits.append({'qid': hit['qid'], 'docid': hit['docid'], 'score': float(hit['score']), 'content': hit['content']})
            self._results.append(Result(query=item['query'], hits=hits))
    
    def get_results(self, with_context: bool):
        if with_context:
            return self._results
        else:
            results = dict()
            for result in self._results:
                query = result.query
                hits = result.hits
                qid = hits[0]['qid']
                results[qid] = dict()
                for hit in hits:
                    pid = hit['docid']
                    score = hit['score']
                    results[qid][pid] = score
            return results