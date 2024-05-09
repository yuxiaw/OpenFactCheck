from core import register_solver, StandardTaskSolver, FactCheckerState
import asyncio
import nest_asyncio
from factool import Factool
from .ftool_utils.chat_api import OpenAIChat
import yaml
import os
from typing import List


##
#
# Factool Decontextualizer
#
# Notes:
#   - This solver is used to extract claims from a response.
#   - The response should be a string.
#
##
@register_solver("factool_decontextualizer", "response", "claims")
class FactoolDecontextualizer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.gpt_model = self.global_config.get("llm_in_use", "gpt-3.5-turbo")
        self.gpt = OpenAIChat(self.gpt_model)
        self.claim_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "ftool_utils/prompts.yaml",
                ),
                "r",
            ),
            yaml.FullLoader,
        )["claim_extraction"]

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        response = state.get(self.input_name)

        claims = self._claim_extraction(responses=[response])[0]

        extracted_claims = [claim["claim"] for claim in claims]

        state.set(self.output_name, extracted_claims)
        return True, state

    def _claim_extraction(self, responses):
        messages_list = [
            [
                {"role": "system", "content": self.claim_prompt["system"]},
                {
                    "role": "user",
                    "content": self.claim_prompt["user"].format(input=response),
                },
            ]
            for response in responses
        ]
        return self.gpt.run(messages_list, List)
