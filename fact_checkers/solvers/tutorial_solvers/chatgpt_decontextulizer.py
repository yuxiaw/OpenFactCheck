from .utils.prompt_base import DECONTEXTILISATION_PROMPT
from .utils.api import chatgpt
from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver


@register_solver("chatgpt_decontextualizer", "sentences", "claims")
class ChatGPTDecontextualizer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        sentences = state.get(self.input_name)

        if sentences is None:
            raise ValueError(f"sentences is required for {self}")

        results = []
        for sentence in sentences:
            user_input = DECONTEXTILISATION_PROMPT + sentence
            decontextualised_claims = chatgpt(user_input)
            print(decontextualised_claims)

            decontextualised_claims = decontextualised_claims.split("\n")
            decontextualised_claims = [claim.strip() for claim in decontextualised_claims if not claim.strip() == ""]
            decontextualised_claims = decontextualised_claims[1:]  # skip 'Output:'
            print("{} decontextualised claims.".format(len(decontextualised_claims)))
            results.extend(decontextualised_claims)

        state.set(self.output_name, results)
        return True, state
