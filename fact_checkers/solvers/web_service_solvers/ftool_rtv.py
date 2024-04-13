from core import register_solver, StandardTaskSolver, FactCheckerState
from .ftool_utils.chat_api import OpenAIChat
from .ftool_utils.search_api import GoogleSerperAPIWrapper
import yaml
import os
from typing import List


@register_solver("factool_retriever", "claims", "claims_with_evidences")
class FactoolRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.gpt_model = self.global_config.get("factool_gpt_model", "gpt-3.5-turbo")
        self.snippet_cnt = args.get("snippet_cnt", 10)
        self.gpt = OpenAIChat(self.gpt_model)
        self.query_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "ftool_utils/prompts.yaml",
                ),
                "r",
            ),
            yaml.FullLoader,
        )["query_generation"]
        self.search_engine = GoogleSerperAPIWrapper(snippet_cnt=self.snippet_cnt)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        queries = self._query_generation(claims=claims)
        evidences = self.search_engine.run(queries)
        results = {}
        for query, claim, evidence in zip(queries, claims, evidences):
            merged_query = ' '.join(query) if len(query) > 1 else str(query)
            results[claim] = [(merged_query, x['content']) for x in evidence]
        state.set(self.output_name, results)
        return True, state

    def _query_generation(self, claims):
        messages_list = [
            [
                {"role": "system", "content": self.query_prompt["system"]},
                {
                    "role": "user",
                    "content": self.query_prompt["user"].format(input=claim),
                },
            ]
            for claim in claims
        ]
        return self.gpt.run(messages_list, List)
