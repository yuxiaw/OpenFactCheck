from core import register_solver, FactCheckerState, StandardTaskSolver
from factool import Factool
import os


##
#
# Factool Solver
#
# Notes:
# Factool requires 3 input parameters: prompt, response, and category.
# Category is always set to 'kbqa' (Knowledge Base Question Answering) for the purposes of this project.
# Because of employing a pipeline of its own, with specific search engine and analysis tools, Factool requires several API keys to be set as environment variables.
# That is:
#   openai_key - OpenAI API key (https://beta.openai.com/)
#   serper_key - Serper API key (https://serper.dev/)
#   scrapper_key - Scrapper API key (https://www.scraperapi.com/)
# Additional parameters:
#   llm_in_use - The OpenAI LLM in use (e.g. gpt-4)
#
##
@register_solver("factool_blackbox", "response", "claim_info")
class FactoolBlackboxSolver(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.input_prompt = args.get("input_prompt", None)
        self.gpt_model = self.global_config.get("llm_in_use", "gpt-4")
        # self.input_prompt = args["input_prompt"] if "input_prompt" in args else None
        # self.gpt_model = args["llm_in_use"] if "llm_in_use" in args else "gpt-4"

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        prompt = state.get(self.input_prompt)
        response = state.get(self.input_name)

        factool_instance = Factool(self.gpt_model)

        inputs = [{"prompt": prompt, "response": response, "category": "kbqa"}]
        claim_info = factool_instance.run(inputs)

        state.set("claim_info", claim_info)
        return True, state
