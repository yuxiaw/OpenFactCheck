# from claim_examiner import ClaimExaminer
from utils.prompt_base import STANCE_DETECTION_PROMPT
from utils.api import chatgpt
import openai
import time
import json
from core.task_solver import StandardTaskSolver
from core.fact_check_state import FactCheckerState
from core import register_solver


@register_solver("chat_gpt_claim_examiner", "evidences", "claim_info")
class ChatGPTClaimExaminer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.path_save_stance = args.get("path_save_stance", "evidence_stance.json")
        self.num_retries = args.get("num_retries", 3)

    def __call__(self, state: FactCheckerState, *args, **kwargs):

        claim_info = state.get("evidences")

        for key, pair in claim_info.items():
            claim = pair['claim']
            evids = pair['evidence_list']
            if len(evids) == 0:
                claim_info[key]["stances"] = []
                continue

            temp = []
            for i, evid in enumerate(evids):
                user_input = STANCE_DETECTION_PROMPT.format(claim, evid["web_page_snippet_manual"])
                for _ in range(self.num_retries):
                    try:
                        stance = chatgpt(user_input)
                        break
                    except openai.OpenAIError as exception:
                        print(f"{exception}. Retrying...")
                        time.sleep(1)
                # print("Claim: {} \n Evidence: {} \n Stance: {}".format(claim, evid, stance))
                evids[i]["stance"] = stance
                temp.append(stance)
            claim_info[key]["stances"] = temp

        # write to json file
        # Serializing json
        json_object = json.dumps(claim_info, indent=4)

        # Writing to sample.json
        with open(self.path_save_stance, "w") as outfile:
            outfile.write(json_object)

        state.set("claim_info", claim_info)
        return True, state
