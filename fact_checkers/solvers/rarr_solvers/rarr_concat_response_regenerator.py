from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver


@register_solver("rarr_concat_response_generator", "revised_claims", "output")
class RARRConcatResponseRegenerator(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        revised_document = " ".join(list(claims.values())).strip()
        # print(revised_document)
        state.set(self.output_name, revised_document)
        return True, state
