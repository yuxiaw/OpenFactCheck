from core import *
from argparse import Namespace
from .utils.openai_api import gpt
from .utils.prompt import CHECKWORTHY_PROMPT_BOOL, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT


@register_solver("checkworthiness_filter", "claims", "claims")
class WorthinessFilter(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        # self.system_role = args.get("system_role", "You are a helpful factchecker assistant.")
        self.system_role = "You are a helpful factchecker assistant."
        self.model = self.global_config.get("model", "gpt-3.5-turbo")
        self.num_retries = self.global_config.get("num_retries", 3)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        results = [True] * len(claims)
        user_input = CHECKWORTHY_PROMPT_BOOL.format(claims=claims)
        response = gpt(user_input, model=self.model, system_role=self.system_role, num_retries=self.num_retries)
        try:
            results = eval(response)
            assert len(results) == len(claims)
        except AssertionError as e:
            print(f"An unexpected error occurred: {e}")
            print(f"There is {len(claims)} texts, while {len(results)} checkworthy predictions.")
            return False, state
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False, state

        state.set(self.output_name, [x for x, y in zip(claims, results) if y])
        return True, state
