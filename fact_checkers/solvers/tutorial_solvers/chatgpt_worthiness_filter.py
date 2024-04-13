from .utils.prompt_base import CHECK_WORTHINESS_LABEL_ONLY_PROMPT
from .utils.api import chatgpt
from typing import List, Tuple
from argparse import Namespace
from core.task_solver import StandardTaskSolver
from core.fact_check_state import FactCheckerState
from core import register_solver


@register_solver("chatgpt_worthiness_filter", "claims", "claims")
class ChatGPTWorthinessFilter(StandardTaskSolver):
    def __init__(self, args: Namespace):
        super().__init__(args)

    # string to format labels
    def convert_checkworthy_output_to_labels(self, label: str) -> bool:
        # factual_labels, checkworthy_labels = [], []
        # for label in labels:
        #
        #     factual_labels.append(opinion_vs_factual)
        #     checkworthy_labels.append(checkworthy)
        #
        # print(factual_labels)
        # print(checkworthy_labels)
        label = label.lower()
        if label[-1] == ".":
            label = label[:-1]
        opinion_vs_factual, checkworthy = label.split(",")
        if "fact" in opinion_vs_factual:
            opinion_vs_factual = "factual"
        else:
            opinion_vs_factual = "opinion"

        if "not" in checkworthy or opinion_vs_factual == "opinion":
            checkworthy = False
        else:
            checkworthy = True
        return checkworthy

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        valid_claims = []
        for claim in claims:
            response = chatgpt(CHECK_WORTHINESS_LABEL_ONLY_PROMPT + claim)
            if self.convert_checkworthy_output_to_labels(response):
                valid_claims.append(claim)

        state.set(self.output_name, valid_claims)
        return True, state
