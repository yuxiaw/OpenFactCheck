import json

class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)

    def get(self, key):
        if key not in self.config:
            return None
        return self.config[key]