import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
from .rarr_utils import agreement_gate, editor, evidence_selection
from .prompts import rarr_prompts
import Levenshtein


@register_solver("rarr_editor", "claims_with_evidences", "revised_claims")
class RARREditor(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("model", "text-davinci-003")
        # self.model = args.get("model", "text-davinci-003")
        self.max_evidences_per_question = args.get("max_evidences_per_question", 1)
        self.max_edit_ratio = args.get("max_edit_ratio", 100)
        self.output_claim_only = args.get("output_claim_only", False)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        final_result = {}
        for claim, contents in claims.items():
            context = contents.get("context", None)
            evidences = contents.get("evidences", [])[:self.max_evidences_per_question]
            agreement_gates = []
            revision_steps = []
            claim_for_iterative_revision = claim
            for evidence in evidences:
                gate = agreement_gate.run_agreement_gate(
                    claim=claim_for_iterative_revision,
                    context=context,
                    query=evidence['query'],
                    evidence=evidence['text'],
                    model=self.model,
                    prompt=rarr_prompts.CONTEXTUAL_AGREEMENT_GATE_PROMPT
                    if context else rarr_prompts.AGREEMENT_GATE_PROMPT
                )
                agreement_gates.append(gate)

                if gate['is_open']:
                    edited_claim = editor.run_rarr_editor(
                        claim=claim_for_iterative_revision,
                        context=context,
                        query=evidence['query'],
                        evidence=evidence['text'],
                        model=self.model,
                        prompt=rarr_prompts.CONTEXTUAL_EDITOR_PROMPT
                        if context
                        else rarr_prompts.EDITOR_PROMPT,
                    )['text']
                    if Levenshtein.distance(claim, edited_claim) / len(claim) <= self.max_edit_ratio:
                        claim_for_iterative_revision = edited_claim
                revision_steps.append({"text": claim_for_iterative_revision})
            result = {
                "context": context,
                "text": claim,
                "questions": contents['questions'],
                "evidences_for_questions": evidences,
                "revisions": [
                    {
                        "original_text": claim,
                        "revised_text": revision_steps[-1]["text"],
                        "evidences": evidences,
                        "agreement_gates": agreement_gates,
                        "revision_steps": revision_steps,
                    }
                ],
            }
            selected_evidences = evidence_selection.select_evidences(result)
            result['selected_evidences'] = selected_evidences
            final_result[claim] = result['revisions'][0]['revised_text'] if self.output_claim_only else result
        state.set(self.output_name, final_result)
        return True, state
