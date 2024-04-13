from flask import Flask
from flask import request
import json
from pipeline import Pipeline
from argparse import ArgumentParser, Namespace
import logging

logger = logging.getLogger(__name__)


def add_args(parser: ArgumentParser):
    parser.add_argument("--config", default='../config/web_service_config_debug.yaml',
                        help="pipline configuration template")
    parser.add_argument("--user_src", default="../solvers/web_service_solvers/", type=str,
                        help="provide user src root with str: './src' or list of str: ['./src','./model_src']")
    parser.add_argument("--output", default='./ws_output', type=str, help="Output path")


app = Flask(__name__)


class LLMFactCheckerWebService:
    def __init__(self, args):
        self.args = args
        self.available_solvers = self.create_service_profile()
        self.pipeline: Pipeline = Pipeline(self.args)
        logger.info("pipline initialized")

    def create_service_profile(self):
        Pipeline.add_user_path(self.args.user_src)
        available_solvers = Pipeline.list_available_solvers()
        config = Pipeline.read_config_from_yaml(self.args.config)
        all_solver_configs = config['all_available_solvers']
        assert len(set(all_solver_configs.keys()) - set(
            available_solvers.keys())) == 0, "Lack available solver config in profile"
        solver_profile = all_solver_configs
        return solver_profile

    def serve(self, *args, **kwargs):
        @app.route("/hi", methods=['GET'])
        def hi():
            return "Hello, I am the LLM Fact Checker Web Service!"

        @app.route("/pipline_info", methods=['GET'])
        def get_pipline_info():
            # return a json list, including the name of solvers used in the current pipeline.
            return json.dumps(self.pipline_info())

        @app.route("/list_solvers", methods=['GET'])
        def list_solvers():
            # Return a json object, with all available solver names and their default configurations
            # Front end needs to save this object in the memory and render their names onto the page
            return json.dumps(self.available_solvers)

        @app.route("/prepare_pipeline", methods=['POST'])
        def prepare_pipeline():
            # Send a json object containing user selected claim processor, retriever and verifier like this:
            '''
            {'factcheck_gpt_claim_processor': 
                {'input_name': 'response',
                  'output_name': 'claims',
                  'mode': 'independent_sentences',
                  'rule_based_method': 'spacy',
                  'spacy_model': 'en_core_web_sm'
                  },
             'factcheck_gpt_retriever': {
                 'input_name': 'claims',
                  'output_name': 'claims_with_evidences',
                  'n_questions': 1,
                  'question_gen_round': 1,
                  'qgen_temp': 0.7,
                  'search_timeout': 10,
                  'max_search_results_per_query': 2,
                  'max_passages_per_search_result_to_return': 3,
                  'sentences_per_passage': 5,
                  'max_passages_per_question': 5,
                  'max_aggregated_evidences': 5
              },
             'factcheck_gpt_verifier': {
                 'input_name': 'claims_with_evidences',
                  'output_name': 'label',
                  'stance_model': 'gpt-3.5-turbo-0613',
                  'verify_retries': 3
                  }}
            '''
            # These configurations stored in the json object returning from /list_solvers
            # Call this API when user makes any change on the pipline.
            config = request.json
            self.pipeline.hot_reload_solvers({"solvers": config}, only_update_solvers=True)
            return json.dumps(self.pipline_info())

        @app.route("/check_factuality", methods=['POST'])
        def check_factuality():
            # Send a json object with a text field in it:
            '''
            {"text": "Alan Turing was a famous physicist who was born in France. He was awarded the Nobel Price in 1953." }
            '''
            user_input = request.json
            return self.check_factuality(user_input['text'])

        app.run(*args, **kwargs)

    def pipline_info(self):
        return list(self.pipeline.solver_pipeline.keys())

    def check_factuality(self, user_input):
        intermediate_result = []

        def record_intermediate(**kwargs):
            intermediate_result.append(kwargs)

        result = self.pipeline(user_input, callback_fun=record_intermediate)

        return json.dumps({"result": result, "intermediate_result": intermediate_result})


'''
Front<->Back end interaction notes
1. Before HTML page ready
    - call /list_solvers, get all available solvers name and configuration
        - Render their names onto the menus for user selection.
        - Save configurations for further usage.
    - call /get_pipline_info, get the solvers used in the current pipeline.
        - Render the names onto menus as "SELECTED" options
2. Page ready
    - If the user use menu to change any components in the pipeline
        - retrieve the configuration of the selected component
        - Create a new config object with order of {CP, RTV, VFR}
        - Send back to server, call /prepare_pipeline to update the pipeline.
    - If the user want to use factuality checking
        - Call /check_factuality with the json payload: {"text", "content in the input box"}
        - Returned object contain two fields
            - result: true/false
            - intermediate_result: json object for each step, content might be different when using different solvers
                Can be used to render onto the page.
'''

if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    llm_fact_checker_service = LLMFactCheckerWebService(args)
    llm_fact_checker_service.serve(host='127.0.0.1', port=8976)
