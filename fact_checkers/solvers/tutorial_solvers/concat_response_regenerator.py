from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver


@register_solver("concat_response_generator", "claim_info", "output")
class ConcatResponseRegenerator(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claim_info = state.get(self.input_name)

        edited_claims = [v["edited_claims"] for _, v in claim_info.items()]
        revised_document = " ".join(edited_claims).strip()
        # print(revised_document)
        state.set(self.output_name, revised_document)
        return True, state
