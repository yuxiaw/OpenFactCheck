from core import register_solver, StandardTaskSolver, FactCheckerState
from .ftool_utils.chat_api import OpenAIChat
import yaml
import os


@register_solver("factool_verifier", "claims_with_evidences", "label")
class FactoolVerifier(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.gpt_model = self.global_config.get("factool_gpt_model", "gpt-3.5-turbo")
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

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        results = self._verification(claims_with_evidences)
        for i, k in enumerate(list(claims_with_evidences.keys())):
            results[i]['claim'] = k
            results[i]['evidences'] = claims_with_evidences[k]
        state.set("detail", results)
        label = all(v['factuality'] for v in results)
        state.set(self.output_name, label)
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
