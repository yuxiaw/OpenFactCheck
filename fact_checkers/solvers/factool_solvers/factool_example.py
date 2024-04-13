from pipeline import Pipeline
from argparse import Namespace

args = Namespace(
    user_src='./factool_solvers/',
    config='./config/factool_config.yaml',
    output='./truth'
)
p = Pipeline(args)
question = "Who is Alan Turing?"
response = "Alan Turing used to be Serbian authoritarian leader, mathematician and computer scientist. He used to be a leader of the French Resistance."
print(p(question=question, response=response))
