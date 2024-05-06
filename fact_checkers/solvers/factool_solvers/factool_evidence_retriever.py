from core import register_solver, StandardTaskSolver, FactCheckerState
from typing import List, Dict, Any
import json
from .ftool_utils.chat_api import OpenAIChat
from .ftool_utils.search_api import GoogleSerperAPIWrapper
import yaml
import os

##
#
# Factool Evidence Retriever
#
# Notes:
#   - This solver is used to retrieve evidences (online content + its sources) for a list of claims.
#   - The claims should be a list of strings.
#   - The evidences are saved in a JSON file.
#
##
@register_solver("factool_evidence_retriever", "claims", "evidences")
class FactoolEvidenceRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.gpt_model = self.global_config.get("llm_in_use", "gpt-4")
        self.gpt = OpenAIChat(self.gpt_model)
        self.path_save_evidence = args.get("path_save_evidence", "evidence.json")
        # self.path_save_evidence = args["path_save_evidence"] if "path_save_evidence" in args else "evidence.json"
        self.queries = None
        self.search_outputs_for_claims = None

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

        self.search_engine = GoogleSerperAPIWrapper(snippet_cnt=10)
        

    # async def coro_queries (self, factool_instance, claims_in_response):
    #    self.queries = await factool_instance.pipelines["kbqa_online"]._query_generation(claims_in_response)
    # async def coro_search_outputs_for_claims (self, factool_instance):
    #    self.search_outputs_for_claims = await factool_instance.pipelines["kbqa_online"].tool.run(self.queries)
    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        queries = self._query_generation(claims=claims)
        search_outputs_for_claims = self.search_engine.run(queries)
        
        
        evidences: Dict[str, Dict[str, Any]] = {}
        for i, claim in enumerate(claims):
            evidence_list: List[dict] = []
            for j, search_outputs_for_claim in enumerate(
                search_outputs_for_claims[i]
            ):
                evidence_list.append(
                    {
                        "evidence_id": j,
                        "web_page_snippet_manual": search_outputs_for_claim["content"],
                        "query": [queries[i]],
                        "url": search_outputs_for_claim["source"],
                        "web_text": [],
                    }
                )
            evidences[claim] = {
                "claim": claim,
                "automatic_queries": queries[i],
                "evidence_list": evidence_list,
            }

        # write to json file
        # Serializing json
        json_object = json.dumps(evidences, indent=4)

        # Writing to sample.json
        with open(self.path_save_evidence, "w") as outfile:
            outfile.write(json_object)

        # print(evidences)

        state.set(self.output_name, evidences)
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
