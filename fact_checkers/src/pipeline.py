import logging
import os
import sys
import tqdm
import json
import yaml
from core.fact_check_state import FactCheckerState
from core import SOLVER_REGISTRY, import_solvers
import traceback

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, args):
        self.args = args

        user_src = getattr(self.args, 'user_src', None)
        if user_src is not None:
            Pipeline.add_user_path(user_src)

        output_path = getattr(self.args, "output", None)
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        else:
            os.makedirs('output', exist_ok=True)
        self.output_path = os.path.abspath(output_path)

        config_file_path = getattr(self.args, "config", None)
        if config_file_path is None or not os.path.isfile(config_file_path):
            return
            # Comment here, make it compatible for web service, supporting multi-stage initialization
            # raise RuntimeError("Invalid config file")
        config_content = Pipeline.read_config_from_yaml(config_file_path)

        self.config, self.global_config = Pipeline.parse_config_from_dict(config_content)

        self.solver_pipeline = self.init_pipeline(self.config)

    def hot_reload_solvers(self, config, only_update_solvers=False):
        if only_update_solvers:
            parsed_config, _ = Pipeline.parse_config_from_dict(config)
            self.config['solvers'] = parsed_config['solvers']
        else:
            self.config, self.global_config = Pipeline.parse_config_from_dict(config)
        self.solver_pipeline = self.init_pipeline(self.config)

    @classmethod
    def add_user_path(cls, user_src):
        try:
            user_srcs = eval(user_src)
            for x in user_srcs:
                logger.info(f"Adding {x} to runtime path...")
                user_src = os.path.abspath(x)
                root_path = os.path.dirname(user_src)
                sys.path.append(root_path)
                namespace = os.path.basename(user_src)
                import_solvers(user_src, namespace)
        except SyntaxError or NameError:
            logger.info(f"Adding {user_src} to runtime path...")
            user_src = os.path.abspath(user_src)
            root_path = os.path.dirname(user_src)
            sys.path.append(root_path)
            namespace = os.path.basename(user_src)
            print("call import solvers with", user_src, namespace)
            import_solvers(user_src, namespace)

    @classmethod
    def list_available_solvers(cls):
        return SOLVER_REGISTRY

    @classmethod
    def parse_config_from_dict(cls, config):
        assert "solvers" in config, "Solvers not found in the config file"
        global_config = dict()
        if "global_config" in config:
            for k, v in config['global_config'].items():
                if type(v) == dict:
                    if 'value' in v:
                        global_config[k] = v['value']
                        if 'env_name' in v:
                            os.environ[v['env_name']] = v['value']
                else:
                    global_config[k] = v
        return config, global_config

    @classmethod
    def read_config_from_yaml(cls, config_file_path):
        config = yaml.load(open(config_file_path, 'r'), yaml.Loader)
        return config

    def init_solver(self, solver_name, args):
        if solver_name not in SOLVER_REGISTRY:
            raise RuntimeError(f"{solver_name} not in SOLVER_REGISTRY")
        solver_cls = SOLVER_REGISTRY[solver_name]
        solver_cls.input_name = args.get("input_name", solver_cls.input_name)
        solver_cls.output_name = args.get("output_name", solver_cls.output_name)
        if len(self.global_config) > 0:
            solver_cls.global_config = self.global_config
        solver = solver_cls(args)
        input_name = solver_cls.input_name
        output_name = solver_cls.output_name
        logger.info(f"Solver {solver} initialized")
        return solver, input_name, output_name

    def init_pipeline(self, solver_config):
        solvers = {}
        for k, v in solver_config['solvers'].items():
            solver, iname, oname = self.init_solver(k, v)
            solvers[k] = (solver, iname, oname)
        return solvers

    def persist_output(self, state: FactCheckerState, idx, solver_name, cont, sample_name=0):
        result = {
            "idx": idx,
            "solver": solver_name,
            "continue": cont,
            "state": state.to_dict()
        }
        with open(os.path.join(self.output_path, f'{sample_name}.jsonl'), 'a', encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def __call__(self, response: str, question: str = None, callback_fun=None, **kwargs):
        sample_name = kwargs.get("sample_name", 0)
        solver_output = FactCheckerState(question=question, response=response)
        oname = "response"
        for idx, (name, (solver, iname, oname)) in tqdm.tqdm(enumerate(self.solver_pipeline.items()),
                                                             total=len(self.solver_pipeline)):
            logger.info(f"Invoking solver: {idx}-{name}")
            logger.debug(f"State content: {solver_output}")
            try:
                solver_input = solver_output
                cont, solver_output = solver(solver_input, **kwargs)
                logger.debug(f"Latest result: {solver_output}")
                if callback_fun:
                    callback_fun(
                        index=idx,
                        sample_name=sample_name,
                        solver_name=name,
                        input_name=iname,
                        output_name=oname,
                        input=solver_input.__dict__,
                        output=solver_output.__dict__,
                        continue_run=cont
                    )
                self.persist_output(solver_output, idx, name, cont, sample_name=sample_name)
            except:
                print(traceback.format_exc())
                cont = False
                oname = iname
            if not cont:
                logger.info(f"Break at {name}")
                break

        return solver_output.get(oname)
