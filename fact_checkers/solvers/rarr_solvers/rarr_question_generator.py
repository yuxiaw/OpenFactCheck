import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
import os
import time
from typing import List
import openai
from .rarr_utils.question_generation import run_rarr_question_generation
from .prompts import rarr_prompts


@register_solver("rarr_question_generator", "claims_with_context", "claims_with_questions")
class RARRQuestionGenerator(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("model", "text-davinci-003")
        self.temperature_qgen = args.get("temperature_qgen", 0.7)
        self.num_rounds_qgen = args.get("num_rounds_qgen", 3)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        # should be DICT[Str, DICT[Str, Any]]
        if type(claims) == list:
            claims = {c: dict() for c in claims}
        for claim, contents in claims.items():
            context = contents.get("context", None)
            claims[claim]['questions'] = run_rarr_question_generation(
                claim=claim,
                context=context,
                model=self.model,
                prompt=rarr_prompts.CONTEXTUAL_QGEN_PROMPT
                if context
                else rarr_prompts.QGEN_PROMPT,
                temperature=self.temperature_qgen,
                num_rounds=self.num_rounds_qgen,
            )

        state.set(self.output_name, claims)
        return True, state
