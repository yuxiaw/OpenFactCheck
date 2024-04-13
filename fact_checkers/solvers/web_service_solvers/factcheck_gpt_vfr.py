from core import *
from argparse import Namespace
from .fc_gpt_utils.prompt import VERIFY_PROMPT
from .fc_gpt_utils.openai_api import gpt
from typing import List, Any, Dict
from .fc_gpt_utils.data_util import save_to_file
from .fc_gpt_utils.prompt import IDENTIFY_STANCE_PROMPT, IDENTIFY_STANCE_PROMPT_FUNC
from .fc_gpt_utils.nli import nli_infer
import pandas as pd
import json


@register_solver("factcheck_gpt_verifier", "claims_with_evidences", "label")
class FactCheckGPTVerifier(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.stance_model = args.get("stance_model", "gpt-3.5-turbo-0613")
        self.num_retries = self.global_config.get("num_retries", 3)
        # self.system_role = args.get("system_role", "You are a helpful factchecker assistant.")
        self.system_role = "You are a helpful factchecker assistant."
        self.verify_retries = args.get("verify_retries", 3)
        self.stance_map = {
            1: "support",
            -1: "refute",
            0: "irrelevant"
        }

    def verify_by_stance(
            self, claim: str,
            evidences: List[str],
    ) -> Any:
        labels = []
        for evidence in evidences:
            labels.append(self.stance(evidence, claim))

        # based on stances of evidence, determine the true/false claim by rules
        # if there is one evidence supports, we assume it is correct
        if 1 in labels:
            return 1
        # if there isn't support, but refute and irrelevant, we regard as false
        elif -1 in labels:
            return -1
        else:
            # all irrelevant
            return 0

    def identify_stance_gpt(self, evidence, claim):
        user_input = IDENTIFY_STANCE_PROMPT_FUNC.format(claim=claim, evidence=evidence)
        r = gpt(
            user_input,
            model=self.stance_model,
            system_role=self.system_role,
            num_retries=self.num_retries
        )
        label = 0
        try:
            label = eval(r)
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
        return label

    def stance(self, evidence, claim, model="gpt-3.5-turbo-0613"):
        """input: a claim and an evidence
           output: label in [support, refute, irrelevant]"""
        label = 0
        if self.stance_model == "nli":
            label = nli_infer(premise=evidence, hypothesis=claim)
        elif "gpt" in self.stance_model:
            label = self.identify_stance_gpt(evidence, claim)
        else:
            print("Check the model argument, choose either gpt or nli model")
        return label

    def verify_claim(self, claim: str, evidences: List[str]) -> Dict[str, Any]:
        results = None
        user_input = VERIFY_PROMPT.format(claim=claim, evidence=evidences)
        r = ''
        for _ in range(self.verify_retries):
            r = gpt(
                user_input,
                model=self.stance_model,
                system_role=self.system_role,
                num_retries=self.num_retries,
            )
            try:
                results = eval(r)
                break
            except Exception as e:
                try:
                    results = json.loads(r)
                except Exception as e:
                    print(f"An unexpected error occurred to parse json {r}: {e}.")
                    save_to_file(r, "verification_error.txt")
                    print(f"An unexpected error occurred to eval {r}: {e}.")

        if isinstance(results, dict):
            return results
        else:
            print(f"Error output {r}. It does not output a dict, return factual label by stance aggregation.")
            factual_label = self.verify_by_stance(claim, evidences)
            results = {
                "reasoning": "",
                "error": "",
                "correction": "",
                "factuality": factual_label
            }
            return results

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        results = []
        for claim, evidences in claims_with_evidences.items():
            result = self.verify_claim(claim, [x[1] for x in evidences])
            result["claim"] = claim
            result["evidences"] = evidences
            results.append(result)
        state.set(self.output_name, all([x['factuality'] > 0 for x in results]))
        state.set("detail", results)
        return True, state
