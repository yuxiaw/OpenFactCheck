from .utils.prompt_base import EDITOR_PROMPT
from .utils.api import chatgpt
import openai
import time
import json
from core.task_solver import StandardTaskSolver
from core.fact_check_state import FactCheckerState
from core import register_solver


@register_solver("chatgpt_post_editor", "claim_info", "claim_info")
class ChatGPTPostEditor(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.path_save_edited_claims = args.get("path_save_edited_claims", "evidence_stance_edit.json")
        self.num_retries = args.get("num_retries", 3)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claim_info = state.get(self.input_name)

        for key, pair in claim_info.items():
            claim = pair['claim'].strip()
            evids = pair['evidence_list']
            stance_explanation = pair['stances']
            # For not checkworthy claims, there is not a list of evidence (stances)
            if len(evids) == 0:
                claim_info[key].set({"edited_claims": claim, "operation": "no-check, no-edit"})
                continue

            # For checkworthy claims, with a list of evidence (stances)
            stance_label = [s.split()[0][:-1].lower() for s in stance_explanation]
            # print(key, stance_label)
            # rules to determine whether and how to edit:
            # if there is one support among stances, claim is regarded as true
            if "support" in stance_label:
                claim_info[key]['edited_claims'] = claim
                claim_info[key]['operation'] = "true, no-edit"
                # claim_info[key].set({"edited_claims": claim, "operation": "true, no-edit"})
            # if all stances are other, not direct/relevant to refute/support the claim, we delete it
            elif all([True for l in stance_label if l == "other"]):
                claim_info[key]['edited_claims'] = ''
                claim_info[key]['operation'] = "no-relevant-evidence, delete"
                # claim_info[key].set({"edited_claims": '', "operation": "no-relevant-evidence, delete"})
            # deal with refute-label with not mention explanation, these evidence is similar to other, just label is ambiguous
            elif all([True for l in stance_explanation if ("other," in l or " not mention" in l)]):
                claim_info[key]['edited_claims'] = ''
                claim_info[key]['operation'] = "no-relevant-evidence, delete"
                # claim_info[key].set({"edited_claims": '', "operation": "no-relevant-evidence, delete"})
            else:
                for i, s in enumerate(stance_label):
                    if s == "other":
                        continue
                    elif s == "refute" and "not mention" in stance_explanation[i]:
                        continue
                    elif s == "refute" or s == "partial support":
                        user_input = EDITOR_PROMPT.format(claim, evids[i])
                        for _ in range(self.num_retries):
                            try:
                                edits = chatgpt(user_input)
                                break
                            except openai.OpenAIError as exception:
                                print(f"{exception}. Retrying...")
                                time.sleep(1)
                        # update claim to revised claim as well
                        claim = edits
                        claim_info[key]["edited_claims"] = edits
                        claim_info[key]["operation"] = "false (refute or partial support), edit"
                        # claim_info[key].set(
                        #     {"edited_claims": edits, "operation": "false (refute or partial support), edit"})
                    else:
                        print(claim)
                        print(s, evids[i])

                        # write to json file
        # Serializing json
        json_object = json.dumps(claim_info, indent=4)

        # Writing to sample.json
        with open(self.path_save_edited_claims, "w") as outfile:
            outfile.write(json_object)

        state.set(self.output_name, claim_info)
        return True, state
