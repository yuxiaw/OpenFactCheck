import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random


# You need to register the solver by giving it a name (which is used in the config),
# and a default input and output name
@register_solver("fake_claim_extractor", "response", "claims")
class FakeClaimExtractor(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.max_claims = args.get("max_claims", 5)
        self.min_claims = args.get("min_claims", 2)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        response = state.get(self.input_name)

        response_len = len(response)
        num_claims = random.randint(min(self.min_claims, response_len), min(self.min_claims, response_len))
        cut_pont = list(range(response_len))
        random.shuffle(cut_pont)
        cut_pont = sorted(cut_pont[:num_claims + 1])
        claims = []
        for i in range(len(cut_pont) - 1):
            claims.append(response[cut_pont[i]:cut_pont[i + 1]])

        state.set(self.output_name, claims)
        return True, state
