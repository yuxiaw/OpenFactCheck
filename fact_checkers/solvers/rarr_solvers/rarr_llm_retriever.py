import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
from .rarr_utils.hallucination import run_evidence_hallucination
from .prompts.hallucination_prompts import EVIDENCE_HALLUCINATION


@register_solver("llm_retriever", "claims_with_questions", "claims_with_evidences")
class LLMRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("model", "text-davinci-003")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        for claim, contents in claims.items():
            questions = contents.get("questions", [])
            evidences = []
            for question in questions:
                evidences.append(
                    run_evidence_hallucination(
                        question,
                        model=self.model,
                        prompt=EVIDENCE_HALLUCINATION
                    )
                )
            claims[claim]['evidences'] = evidences

        state.set(self.output_name, claims)
        return True, state
