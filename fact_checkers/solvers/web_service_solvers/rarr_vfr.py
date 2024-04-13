import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
from .rarr_utils import agreement_gate
from .rarr_prompts import functional_prompt


@register_solver("rarr_verifier", "claims_with_evidences", "label")
class RARRAgreementGate(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.max_evidences_per_question = args.get("max_evidences_per_question", 1)
        self.model = self.global_config.get("rarr_model", "text-davinci-003")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        results = []
        for claim, evidences in claims_with_evidences.items():
            result = {}
            evidences = evidences[:self.max_evidences_per_question]
            labels = []
            for query, evidence in evidences:
                gate = agreement_gate.run_agreement_gate(
                    claim=claim,
                    context=None,
                    query=query,
                    evidence=evidence,
                    model=self.model,
                    prompt=functional_prompt.AGREEMENT_GATE_PROMPT
                )
                labels.append(gate['is_open'])
            result['claim'] = claim
            result['evidences'] = evidences
            result['labels'] = labels
            result['factuality'] = all(labels)
            results.append(result)
        state.set(self.output_name, all([x['factuality'] for x in results]))
        state.set("detail", results)
        return True, state
