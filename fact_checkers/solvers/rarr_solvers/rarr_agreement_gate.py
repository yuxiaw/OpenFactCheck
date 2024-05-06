import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
from .rarr_utils import agreement_gate
from .prompts import rarr_prompts


@register_solver("rarr_agreement_gate", "claims_with_evidences", "claims_with_gates")
class RARRAgreementGate(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.max_evidences_per_question = args.get("max_evidences_per_question", 1)
        self.model = self.global_config.get("model", "text-davinci-003")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        for claim, contents in claims.items():
            context = contents.get("context", None)
            evidences = contents.get("evidences", [])[:self.max_evidences_per_question]
            gates = []
            for evidence in evidences:
                gate = agreement_gate.run_agreement_gate(
                    claim=claim,
                    context=context,
                    query=evidence['query'],
                    evidence=evidence['text'],
                    model=self.model,
                    prompt=rarr_prompts.CONTEXTUAL_AGREEMENT_GATE_PROMPT
                    if context else rarr_prompts.AGREEMENT_GATE_PROMPT
                )
                gates.append(gate)
            contents['gates'] = gates

        state.set(self.output_name, claims)
        return True, state
