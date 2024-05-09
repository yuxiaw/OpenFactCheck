import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from src.service import FactCheckService
from cmn.config import Config

# Initialize the Flask app
app = Flask(__name__)

# Initialize the FactCheckService only when it's needed and the config is already set
factcheck_service = None

def initialize_service():
    global factcheck_service
    config_path = app.config.get('CONFIG_PATH')
    if config_path:
        factcheck_service = FactCheckService(Config(config_path))

@app.route("/health", methods=['GET'])
def health():
    return "Hello, I am the LLM Fact Checker Web Service!"

@app.route("/pipeline/configuration", methods=['GET'])
def pipeline_configuration():
    return jsonify(factcheck_service.pipline_configuration())

@app.route("/pipeline/solver", methods=['GET'])
def list_solvers():
    return jsonify(factcheck_service.pipeline_solvers())

@app.route("/pipeline/solver/claimprocessor", methods=['GET'])
def list_claim_processors():
    solvers = factcheck_service.pipeline_solvers()

    # Get all claim processors
    claim_processors = {}
    for solver, value in solvers.items():
        if "claim_processor" in solver:
            claim_processors[solver] = value

    return jsonify(claim_processors)

@app.route("/pipeline/solver/retriever", methods=['GET'])
def list_retrievers():
    solvers = factcheck_service.pipeline_solvers()

    # Get all retrievers
    retrievers = {}
    for solver, value in solvers.items():
        if "retriever" in solver:
            retrievers[solver] = value

    return jsonify(retrievers)

@app.route("/pipeline/solver/verifier", methods=['GET'])
def list_verifiers():
    solvers = factcheck_service.pipeline_solvers()

    # Get all verifiers
    verifiers = {}
    for solver, value in solvers.items():
        if "verifier" in solver:
            verifiers[solver] = value

    return jsonify(verifiers)

@app.route("/pipeline/configure/global", methods=['POST'])
def configure_pipeline_global():
    config = request.json
    return jsonify(factcheck_service.pipeline_configure_global(config))

@app.route("/pipeline/configure/solvers", methods=['POST'])
def configure_pipeline_solvers():
    config = request.json
    return jsonify(factcheck_service.pipeline_configure_solvers(config))

@app.route("/evaluate/response", methods=['POST'])
def evaluate_response():
    user_input = request.json
    return jsonify(factcheck_service.evaluate_response(user_input['text']))

@app.route("/evaluate/llm/<id>", methods=['POST'])
def evaluate_llm(id):
    return jsonify(factcheck_service.evaluate_llm(id))

@app.route("/evaluate/factchecker/<id>", methods=['POST'])
def evaluate_factchecker(id):
    return jsonify(factcheck_service.evaluate_factchecker(id))
