import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string


@register_solver("confused_claim_examiner", "claims_with_evidences", "claims_with_tags")
class ConfusedClaimExaminer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        claims_with_tags = {}
        for claim_key, docs in claims.items():
            claims_with_tags[claim_key] = random.choice([True, False])

        state.set(self.output_name, claims_with_tags)
        return True, state
