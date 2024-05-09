from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver
from core import register_solver
from .ftool_utils.chat_api import OpenAIChat
import yaml
import os
import json


##
#
# Factool Claim Examiner
#
# Notes:
#   - This solver is used to examine the claims in a response.
#
##
@register_solver("factool_claim_examiner", "evidences", "claim_info")
class FactoolClaimExaminer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.gpt_model = self.global_config.get("llm_in_use", "gpt-4")
        self.path_save_stance = args.get("path_save_stance", "evidence_stance.json")
        self.verifications = None
        self.gpt = OpenAIChat(self.gpt_model)
        self.verification_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "ftool_utils/prompts.yaml",
                ),
                "r",
            ),
            yaml.FullLoader,
        )["verification"]

    # async def coro (self, factool_instance, claims_in_response, evidences):
    #    self.verifications = await factool_instance.pipelines["kbqa_online"]._verification(claims_in_response, evidences)
    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claim_info = state.get(self.input_name)
        # Recover the Factool objects
        claims_in_response = []
        queires = []
        search_outputs_for_claims = []
        for key, pair in claim_info.items():
            claim = key or pair["claim"]
            claims_in_response.append({"claim": claim})
            queires.append(pair["automatic_queries"])
            search_outputs_for_claim = []
            for evidence in pair["evidence_list"]:
                search_outputs_for_claim.append(
                    {
                        "content": evidence["web_page_snippet_manual"],
                        "source": evidence["url"],
                    }
                )
            search_outputs_for_claims.append(search_outputs_for_claim)

        claims_with_evidences = {k: [u['web_page_snippet_manual'] for u in claim_info[k]['evidence_list']] for k in
                                 claim_info.keys()}
        verifications = self._verification(claims_with_evidences)

        # evidences = [
        #     [output["content"] for output in search_outputs_for_claim]
        #     for search_outputs_for_claim in search_outputs_for_claims
        # ]

        # Attach the verifications (stances) to the claim_info
        for index, (key, pair) in enumerate(claim_info.items()):
            # print(f'Verifications: {verifications}\n')
            # print(f'Verification for claim {key}: Index {index}\n')
            # print(f'Verification for claim {key}: {verifications[index]}\n')
            # print(f'Verification for claim {key}: Type = {type(verifications[index])}\n')
            stance = ""
            if (
                    type(verifications[index]) == None
                    or verifications[index] == "None"
            ):
                stance = claims_in_response[index]["claim"]
            else:
                stance = (
                    ""
                    if (
                            verifications[index]["error"] == "None"
                            or len(verifications[index]["error"]) == 0
                    )
                    else (verifications[index]["error"] + " ")
                )
                stance += (
                    ""
                    if (
                            verifications[index]["reasoning"] == "None"
                            or len(verifications[index]["reasoning"]) == 0
                    )
                    else verifications[index]["reasoning"]
                )
                stance += (
                    claims_in_response[index]["claim"]
                    if (
                            verifications[index]["correction"] == "None"
                            or len(verifications[index]["correction"]) == 0
                    )
                    else (" " + verifications[index]["correction"])
                )
            claim_info[key]["stances"] = [stance]
            for j in range(len(claim_info[key]["evidence_list"])):
                claim_info[key]["evidence_list"][j]["stance"] = stance

        # write to json file
        # Serializing json
        json_object = json.dumps(claim_info, indent=4)

        # Writing to sample.json
        with open(self.path_save_stance, "w") as outfile:
            outfile.write(json_object)

        # print(claim_info)

        state.set(self.output_name, claim_info)
        return True, state

    def _verification(self, claims_with_evidences):
        messages_list = [
            [
                {"role": "system", "content": self.verification_prompt['system']},
                {"role": "user", "content": self.verification_prompt['user'].format(claim=claim, evidence=str(
                    [e[1] for e in evidence]))},
            ]
            for claim, evidence in claims_with_evidences.items()
        ]
        return self.gpt.run(messages_list, dict)
