import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string


@register_solver("useless_response_regenerator", "claims_with_tags", "output")
class UselessResponseRegenerator(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        true_claims = [k[1] for k, v in claims.items() if v is True]
        new_response = ' '.join(true_claims)
        state.set(self.output_name, new_response)
        return True, state
