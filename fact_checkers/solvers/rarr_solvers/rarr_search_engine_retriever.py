import logging

from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
import random
import string
from .rarr_utils import search


@register_solver("search_engine_retriever", "claims_with_questions", "claims_with_evidences")
class SearchEngineRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.max_search_results_per_query = args.get("max_search_results_per_query", 5)
        self.max_sentences_per_passage = args.get("max_sentences_per_passage", 4)
        self.sliding_distance = args.get("sliding_distance", 1)
        self.max_passages_per_search_result = args.get("max_passages_per_search_result", 1)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)

        for claim, contents in claims.items():
            questions = contents.get("questions", [])
            evidences = []
            for question in questions:
                evidences.extend(
                    search.run_search(
                        query=question,
                        max_search_results_per_query=self.max_search_results_per_query,
                        max_sentences_per_passage=self.max_sentences_per_passage,
                        sliding_distance=self.sliding_distance,
                        max_passages_per_search_result_to_return=self.max_passages_per_search_result,
                    )
                )
            claims[claim]['evidences'] = evidences

        state.set(self.output_name, claims)
        return True, state
