from core import register_solver, StandardTaskSolver, FactCheckerState
import nltk
import spacy
from .fc_gpt_utils.openai_api import gpt
from .fc_gpt_utils.data_util import save_to_file
from .fc_gpt_utils.prompt import DOC_TO_INDEPEDENT_SENTENCES_PROMPT, SENTENCES_TO_CLAIMS_PROMPT, \
    DOC_TO_SENTENCES_PROMPT, CHECKWORTHY_PROMPT_BOOL, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT
from argparse import Namespace


@register_solver("factcheck_gpt_claim_processor", "response", "claims")
class FactCheckGPTClaimProcessor(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("factcheck_gpt_model", "gpt-3.5-turbo")
        self.num_retries = self.global_config.get("num_retries", 3)
        self.mode = args.get("mode", "independent_sentences")
        self.decompose_system_role = "You are good at decomposing and decontextualizing text."
        self.worthines_filter_system_role = "You are a helpful factchecker assistant."
        self.rule_based_method = args.get("rule_based_tool", "spacy")
        self.spacy_model = args.get("spacy_model", "en_core_web_sm")
        self.prompt = {
            "sentences": DOC_TO_SENTENCES_PROMPT,
            "independent_sentences": DOC_TO_INDEPEDENT_SENTENCES_PROMPT,
            "claims": SENTENCES_TO_CLAIMS_PROMPT
        }.get(self.mode, DOC_TO_INDEPEDENT_SENTENCES_PROMPT)
        nlp = spacy.load(self.spacy_model)
        self.rule_based_tool = {
            "nltk": lambda x: [x.strip() for x in nltk.sent_tokenize(x) if len(x.strip()) >= 3],
            "spacy": lambda x: [x.text.strip() for x in nlp(x).sents if len(x.text.strip()) >= 3]
        }.get(self.rule_based_method, "nltk")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        # We have merged the text decomposer and worthiness filter here.
        response = state.get(self.input_name)
        claims = [response]
        
        user_input = self.prompt.format(doc=response).strip()
        r = gpt(user_input, model=self.model, system_role=self.decompose_system_role, num_retries=self.num_retries)
        try:
            claims = eval(r)
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            save_to_file(r)

        if not isinstance(claims, list):
            print(
                f"{self.model} output {r}. It does not output a list of sentences correctly, return rule-based split results.")
            claims = self.rule_based_tool(response)
            
        worthiness = [True] * len(claims)
        user_input = CHECKWORTHY_PROMPT_BOOL.format(claims=claims)
        response = gpt(user_input, model=self.model, system_role=self.worthines_filter_system_role,
                       num_retries=self.num_retries)
        # TODO refine check worthiness prompt, value returned not reasonable.
        try:
            worthiness = eval(response)
            assert len(worthiness) == len(claims)
        except AssertionError as e:
            print(f"An unexpected error occurred: {e}")
            print(f"There are {len(claims)} texts, while {len(worthiness)} checkworthy predictions.")
            return False, state
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False, state

        valid_claims = list(map(lambda x: x[1], filter(lambda x: x[0], zip(worthiness, claims))))
        state.set(self.output_name, valid_claims)
        return True, state
