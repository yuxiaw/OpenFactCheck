import sys

from .fact_check_state import *
from .task_solver import *
import os
import importlib
import glob
import logging

SOLVER_REGISTRY = {}
logger = logging.getLogger(__name__)


def register_solver(name, input_name=None, output_name=None):
    def register_solver_cls(cls):
        if name in SOLVER_REGISTRY:
            return SOLVER_REGISTRY[name]

        if not issubclass(cls, StandardTaskSolver):
            raise ValueError(
                "Solver ({}: {}) must extend StandardTaskSolver".format(name, cls.__name__)
            )
        SOLVER_REGISTRY[name] = cls
        cls.name = name
        cls.input_name = input_name
        cls.output_name = output_name
        return cls

    return register_solver_cls


def import_solvers(solver_dir, namespace):
    print(solver_dir, namespace)
    if os.path.isdir(solver_dir):
        for file in sorted(
                os.listdir(solver_dir),
                key=lambda x: int(os.path.isdir(os.path.join(solver_dir, x))),
                reverse=True
        ):
            if (
                    not file.startswith('_') and not file.startswith('.')
            ):
                # print(os.path.abspath(file))
                next_node = os.path.join(solver_dir, file)
                if os.path.isdir(next_node):
                    import_solvers(os.path.join(solver_dir, file), namespace + '.' + file)
                else:
                    import_solvers(os.path.join(solver_dir, file), namespace)
    else:
        file = os.path.basename(solver_dir)
        path = solver_dir
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            solver_name = file[: file.find(".py")] if file.endswith(".py") else file
            print("Importing", namespace + "." + solver_name)
            importlib.import_module(namespace + "." + solver_name)

    return
# 
# solvers_dir = os.path.dirname(__file__)
# namespace = os.path.basename(solvers_dir)
# import_solvers(solvers_dir, namespace)
# logger.info(f"Solvers loaded: {SOLVER_REGISTRY}")
