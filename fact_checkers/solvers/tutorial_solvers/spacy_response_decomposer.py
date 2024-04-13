import logging

import spacy
from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver


@register_solver("spacy_response_decomposer", 'response', 'sentences')
class SpacyResponseDecomposer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        spacy_model = args.get("spacy_model", "en_core_web_sm")
        self.spacy_processor = spacy.load(spacy_model)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        response = state.get(self.input_name)
        doc = self.spacy_processor(response)
        sentences = [str(sent).strip() for sent in doc.sents]
        logging.info("The document is split into {} sentences.".format(len(sentences)))
        state.set(self.output_name, sentences)
        return True, state
