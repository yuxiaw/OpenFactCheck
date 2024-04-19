import os
import time
import threading
import subprocess
from cmn.config import Config
from pipeline import Pipeline
from argparse import Namespace

class FactCheckService:
    def __init__(self, config: Config):
        self.config = config
        pipline_config = {"usr_src": self.config.get('pipeline').get('solvers_path'), 
                          "config": self.config.get('pipeline').get('config_path'), 
                          "output": self.config.get('pipeline').get('output_path')}
        
        # Get all available solvers
        self.available_solvers = self.pipeline_solvers()

        # Initialize the pipeline
        self.pipeline = Pipeline(Namespace(**pipline_config))
        print("FactCheckService initialized")

        # Initialize the process pool
        self.process_pool = {}
        cleaning_thread = threading.Thread(target=self.llm_process_clean)
        cleaning_thread.start()

    def pipline_configuration(self):
        return list(self.pipeline.solver_pipeline.keys())
    
    def pipline_configure(self, config):
        self.pipeline.hot_reload_solvers({"solvers": config}, only_update_solvers=True)
        return self.pipline_configuration()

    def pipeline_solvers(self):
        # Add solvers path to the runtime path
        Pipeline.add_user_path(self.config.get('pipeline').get('solvers_path'))

        # Get all available solvers
        available_solvers = Pipeline.list_available_solvers()

        # Get default configuration for the pipeline
        config = Pipeline.read_config_from_yaml(self.config.get('pipeline').get('config_path'))

        # Get all available solvers
        solver_profile = config['all_available_solvers']

        # Check if all available solvers are in the profile
        assert len(set(solver_profile.keys()) - set(available_solvers.keys())) == 0, "Lack available solver config in profile"

        # Return the solver profile
        return solver_profile

    def evaluate_response(self, user_input):
        # Record intermediate results
        intermediate_result = []

        # Callback function to record intermediate results
        def record_intermediate(**kwargs):
            intermediate_result.append(kwargs)

        # Evaluate the response
        result = self.pipeline(user_input, callback_fun=record_intermediate)

        # Return the result and intermediate results
        return {"result": result, "intermediate_result": intermediate_result}
    
    def evaluate_llm(self, id):
        script_path = self.config.get("llm").get("evaluator_path")
        filename = f"../eval_results/llm/{id}/input/response.csv"

        # Check if the file exists
        if not os.path.exists(filename):
            return {"error": "File not found"}
        
        # cd to the script path
        os.chdir(os.path.dirname(script_path))
    
        process = subprocess.Popen(
            f"bash process.sh {filename} {id}".split()
        )

        self.process_pool[process.pid] = process

        return {"job_id": id}
    
    def evaluate_factchecker(self, id):
        script_path = self.config.get("factchecker").get("evaluator_path")
        filename = f"../eval_results/factchecker/{id}/input/response.csv"

        # Check if the file exists
        if not os.path.exists(filename):
            return {"error": "File not found"}
        
        # cd to the script path
        os.chdir(os.path.dirname(script_path))
    
        process = subprocess.Popen(
            f"bash process.sh {filename} {id}".split()
        )

        self.process_pool[process.pid] = process

        return {"job_id": id}
    
    def llm_process_clean(self):
        while True:
            for k in list(self.process_pool.keys()):
                if self.process_pool[k].poll() is not None:
                    del self.process_pool[k]
            time.sleep(60)