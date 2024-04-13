import json
import logging
import pprint


class FactCheckerState:
    def __init__(self, question: str = None, response: str = None):
        self.question: str = question
        self.response: str = response

    def set(self, name, value):
        if hasattr(self, name):
            logging.warning(f"modifying existing attribute: {name}")
        setattr(self, name, value)

    def get(self, name):
        if not hasattr(self, name):
            raise ValueError(f"{name} not in the state")
        return getattr(self, name, None)

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__
