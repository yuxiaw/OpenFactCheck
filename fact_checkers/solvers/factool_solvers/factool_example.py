import os
from pipeline import Pipeline
from argparse import Namespace

# Base directory where the script is located
base_dir = os.path.abspath(os.path.dirname(__file__))

args = Namespace(
    user_src=os.path.join(base_dir),
    config=os.path.join(base_dir, "../../config/factool_config.yaml"),
    output=os.path.join(base_dir, "../../../output")
)

p = Pipeline(args)
question = "Who is Alan Turing?"
response = "Alan Turing was a British mathematician, logician, cryptanalyst, and computer scientist. He was highly influential in the development of theoretical computer science."

print(p(question=question, response=response))
