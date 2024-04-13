from .fact_check_state import FactCheckerState
from typing import Tuple
import logging


class StandardTaskSolver:
    name: str = None
    input_name: str = None
    output_name: str = None
    global_config: dict = dict()

    def __init__(self, args: dict):
        self.args = args
        logging.debug(self.args)

    def __call__(self, state: FactCheckerState, **kwargs) -> Tuple[
        bool, FactCheckerState]:
        raise NotImplementedError

    @classmethod
    def build_solver(cls, args):
        raise NotImplementedError

    @property
    def input_name(self):
        return self.__class__.input_name

    @property
    def output_name(self):
        return self.__class__.output_name

    def __str__(self):
        return f'[name:"{self.__class__.name}", input: "{self.__class__.input_name}": output: "{self.__class__.output_name}"]'
