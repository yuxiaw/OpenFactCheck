# Load evaluate_llm_factuality_dataset.jsonl
# Run factool over each response, save all intermediate results (important)
# Calculate the ratio of true claims, and true responses
# Calculate time, costs as well

###### ***TODO:*** *Update the exact matching (EM) with data from numUndefinedClaims in the evidence result files. Debug the few missing evidence result experiments (this means debugging and patching the Google scraper module in the original Factool).* ######

###
# Import modules and libraries logic
###

import pandas as pd
import os
import sys
import time
import json
import math
from hashlib import md5
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

# Add the llm-fact-checker pipeline and factool to the system path
pipeline_dir = os.path.join(os.path.dirname(os.getcwd()), 'fact_checkers', 'src')
# factool_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'factool-main', 'factool')
sys.path.insert(0, pipeline_dir)
# sys.path.insert(0, factool_dir)

from pipeline import Pipeline
# from factool import Factool
import os

###
# Project settings
###

# Folder to save the results
projectdir = "factool_benchmark"
# os.makedirs(projectdir, exist_ok=True)
# LLM responses of interest
response_types = ["GPT4_response", "llama7b_response", "llama13b_response"]
# Solver arguments for the factool pipeline (any other pipeline could be integrated in the exact or similar manner)
examples_dir = os.path.join(os.path.dirname(os.getcwd()), 'fact_checkers', 'solvers')
solver_args = Namespace(
    user_src=os.path.join(examples_dir, 'factool_solvers'),
    config=os.path.join(examples_dir, 'config', 'factool_config.yaml'),
    output='./truth'
)


###
# Helper methods
###

# Calculate cost (in USD) for the API calls => 2x API calls per claim
def calcPrice(numClaims, costOpenAI=0.015, costSerper=0.001):
    return numClaims * 2 * (costOpenAI + costSerper)


# Sum all elements of an object
def sumAllObj(obj):
    ret = 0
    for k, v in obj.items():
        ret += v
    return ret


##
# Example cost calculation from a datadir entry
##
# Sum all claims, no matter their stance
# sumClaims=sumAllObj(d["claims"])
# Calculate cost
# cost=calcPrice(sumClaims)

# Collect data from project directory
def collect_data(projectdir=projectdir):
    data = []
    for dirname in os.listdir(projectdir):
        dirpath = os.path.join(projectdir, dirname)
        if os.path.isdir(dirpath):
            if os.path.exists(os.path.join(dirpath, 'eval_result.json')):
                with open(os.path.join(dirpath, 'eval_result.json'), 'r') as f:
                    data.append(json.load(f))
    return data


###
# Data extraction and analysis methods
###

# Read the dataset and select the required responses
def read_llm_responses(filename='evaluate_llm_factuality_dataset.jsonl', projectdir='./'):
    path = os.path.join(projectdir, filename)
    df = pd.read_json(path, lines=True)
    # Filter only the ones with source=factool-qa, source=felm-wk, source=factcheckgpt
    df = df[df["source"].str.contains('factool-qa|felm-wk|factcheckgpt')]
    # Extract prompt, GPT4_response, llama7b_response, llama13b_response
    data = {
        "source": list(df["source"]),
        "prompt": list(df["prompt"]),
        "GPT4_response": list(df["GPT4_response"]),
        "llama7b_response": list(df["llama7b_response"]),
        "llama13b_response": list(df["llama13b_response"])
    }
    print("Items: ", len(data["source"]))

    return data


# Extract number and type of claims => that is, Exact Matching (EM)
def assess_experiment():
    ret = {
        "numFalseClaims": 0,
        "numMixedClaims": 0,
        "numTrueClaims": 0,
        "numUndefinedClaims": 0
    }
    path = 'evidence_stance.json'
    if not os.path.exists(path):
        return False
    df = pd.read_json(path, lines=False)
    dataobj = json.loads(df.to_json())
    for k, v in dataobj.items():
        # If stance contains definitive or mixed, then it is false
        if "definitive" in v["stances"][0] or "mixed" in v["stances"][0]:
            ret["numMixedClaims"] += 1
        elif "factual" in v["stances"][0] or "confirm" in v["stances"][0]:
            ret["numTrueClaims"] += 1
        elif "error" in v["stances"][0] or "incorrect" in v["stances"][0] or "false" in v["stances"][0]:
            ret["numFalseClaims"] += 1
        else:
            ret["numUndefinedClaims"] += 1
            print(k, ':', v['stances'][0], '\n')
    return ret


# Execute the factool pipeline solvers
def execute_factool_solvers(args=solver_args, projectdir=projectdir):
    ###
    #
    # Check all combinations of prompts and responses
    #
    # Output paths for intermediate data are based on the current prompt index and on the md5 hash of the prompt
    # e.g. for assessing GPT4 response on a prompt, prompt being "Question: Tell me a bio of Lanny Flaherty." with index 0,
    # the output path would be ./GPT4_0_5e6f4bd525af0865f7ad98ae43c75071. Suggestion: we could easily optimize this using Redis.
    #
    ###
    llm_response_data = read_llm_responses()
    p = Pipeline(args)

    if "openai_apikey" in args:
        p.hot_reload_global_config({"global_config": {"openai_key": {"value": args.openai_apikey, "env_name": "OPENAI_API_KEY"}}})
    
    for i in range(0, len(llm_response_data["prompt"])):
        prompt = llm_response_data["prompt"][i]
        # reponses = [llm_response_data[response_type][i] for response_type in response_types]
        current_path = os.getcwd()
        for response_type in response_types:
            response = llm_response_data[response_type][i]
            # print(response_type)
            os.chdir(current_path)
            # Remove _response
            dirname = response_type[:-9] + "_" + str(i) + "_" + md5(prompt.encode()).hexdigest()
            dirpath = os.path.join(projectdir, dirname)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            os.chdir(dirpath)
            # Keep the intermediate data in dirpath. Calculate the ratio of true claims, and true responses. Calculate time, costs as well
            start = time.time() * 1000
            _result = p(question=prompt, response=response, sample_name=f"{i}_{response_type}")
            end = time.time() * 1000
            # Process the result data
            claims = assess_experiment()
            if claims == False:
                print(f'\n\nError in assessing experiment for prompt {i} and response {response_type}\n\n')
                os.chdir(current_path)
                continue
            # Keep the results
            result = {}
            result["start"] = math.floor(start)
            result["end"] = math.floor(end)
            result["llm"] = response_type
            result["dataset"] = llm_response_data["source"][i]
            result["prompt"] = prompt
            result["claims"] = claims
            result["result"] = _result
            print(result)
            # Write to json file
            with open('eval_result.json', 'w') as f:
                json.dump(result, f)
            # print(result)
            print("Done with <", response_type, "> on prompt [", i, "] '", prompt, "' in ", end - start, " ms.")
            # Just in case
            os.chdir(current_path)
    print(f'\n\n *** Done with all prompts and responses ***\n\n')


def evaluate_free_text_by_factool(llm_response_data, response_column_name, 
                                  args=solver_args, projectdir=projectdir):
    ###
    #
    # Check all combinations of prompts and responses
    #
    # Output paths for intermediate data are based on the current prompt index and on the md5 hash of the prompt
    # e.g. for assessing GPT4 response on a prompt, prompt being "Question: Tell me a bio of Lanny Flaherty." with index 0,
    # the output path would be ./GPT4_0_5e6f4bd525af0865f7ad98ae43c75071. Suggestion: we could easily optimize this using Redis.
    #
    ###
    dataset_name = llm_response_data[0]['source']
    llm_response_data = pd.DataFrame(llm_response_data)

    p = Pipeline(args)

    if "openai_apikey" in args:
        p.hot_reload_global_config({"global_config": {"openai_key": {"value": args.openai_apikey, "env_name": "OPENAI_API_KEY"}}})
    
    for i in range(0, len(llm_response_data["prompt"])):
        prompt = llm_response_data["prompt"][i]
        current_path = os.getcwd()
        response_type = response_column_name
        response = llm_response_data["response"][i]
        os.chdir(current_path)
        # Remove _response
        dirname = response_type[:-9] + f"_{dataset_name}_" + str(i) + "_" + md5(prompt.encode()).hexdigest()
        dirpath = os.path.join(projectdir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        os.chdir(dirpath)
        # Keep the intermediate data in dirpath. Calculate the ratio of true claims, and true responses. Calculate time, costs as well
        start = time.time() * 1000
        _result = p(question=prompt, response=response, 
                    sample_name=response_type[:-9] + f"_{dataset_name}_" + str(i))
        end = time.time() * 1000
        # Process the result data
        claims = assess_experiment()
        if claims == False:
            print(f'\n\nError in assessing experiment for prompt {i} and response {response_type}\n\n')
            os.chdir(current_path)
            continue
        # Keep the results
        result = {}
        result["start"] = math.floor(start)
        result["end"] = math.floor(end)
        result["llm"] = response_type
        result["dataset"] = llm_response_data["source"][i]
        result["prompt"] = prompt
        result["claims"] = claims
        result["result"] = _result
        print(result)
        # Write to json file
        with open('eval_result.json', 'w') as f:
            json.dump(result, f)
        # print(result)
        print("Done with <", response_type, "> on prompt [", i, "] '", prompt, "' in ", end - start, " ms.")
        # Just in case
        os.chdir(current_path)
    print(f'\n\n *** Done with all prompts and responses ***\n\n')


###
# Charts
###

###### Grouped barplot for false claims per dataset per LLM ######
def gr_barplot_false_claims():
    data = collect_data()
    # Filter factool-qa|felm-wk|factcheckgpt datasets and GPT4|llama7b|llama13b llms
    falseClaims = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for d in data:
        if d["llm"] == "GPT4_response":
            if d["dataset"] == "factool-qa":
                falseClaims[0][0] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "felm-wk":
                falseClaims[0][1] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "factcheckgpt":
                falseClaims[0][2] += d["claims"]["numFalseClaims"]
        elif d["llm"] == "llama7b_response":
            if d["dataset"] == "factool-qa":
                falseClaims[1][0] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "felm-wk":
                falseClaims[1][1] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "factcheckgpt":
                falseClaims[1][2] += d["claims"]["numFalseClaims"]
        elif d["llm"] == "llama13b_response":
            if d["dataset"] == "factool-qa":
                falseClaims[2][0] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "felm-wk":
                falseClaims[2][1] += d["claims"]["numFalseClaims"]
            elif d["dataset"] == "factcheckgpt":
                falseClaims[2][2] += d["claims"]["numFalseClaims"]
    # set width of bars
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(falseClaims[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    plt.bar(r1, falseClaims[0], color='#A569BD', width=barWidth, edgecolor='white', label='GPT-4')
    plt.bar(r2, falseClaims[1], color='#453896', width=barWidth, edgecolor='white', label='Llama-7b')
    plt.bar(r3, falseClaims[2], color='#6790D7', width=barWidth, edgecolor='white', label='Llama-13b')
    # Add xticks on the middle of the group bars
    plt.xlabel('False claims per dataset per LLM', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(falseClaims[0]))], ['factool-qa', 'felm-wk', 'factcheckgpt'])
    plt.ylabel('Number of false claims', fontweight='bold')
    # Create legend & Show graphic
    plt.legend()
    plt.show()


###### Grouped barplot for costs per dataset per LLM ######
def gr_barplot_costs():
    data = collect_data()
    # Filter factool-qa|felm-wk|factcheckgpt datasets and GPT4|llama7b|llama13b llms
    costs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for d in data:
        if d["llm"] == "GPT4_response":
            if d["dataset"] == "factool-qa":
                costs[0][0] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "felm-wk":
                costs[0][1] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "factcheckgpt":
                costs[0][2] += calcPrice(sumAllObj(d["claims"]))
        elif d["llm"] == "llama7b_response":
            if d["dataset"] == "factool-qa":
                costs[1][0] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "felm-wk":
                costs[1][1] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "factcheckgpt":
                costs[1][2] += calcPrice(sumAllObj(d["claims"]))
        elif d["llm"] == "llama13b_response":
            if d["dataset"] == "factool-qa":
                costs[2][0] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "felm-wk":
                costs[2][1] += calcPrice(sumAllObj(d["claims"]))
            elif d["dataset"] == "factcheckgpt":
                costs[2][2] += calcPrice(sumAllObj(d["claims"]))
    # set width of bars
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(costs[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    plt.bar(r1, costs[0], color='#A569FF', width=barWidth, edgecolor='white', label='GPT-4')
    plt.bar(r2, costs[1], color='#4538FF', width=barWidth, edgecolor='white', label='Llama-7b')
    plt.bar(r3, costs[2], color='#6790FF', width=barWidth, edgecolor='white', label='Llama-13b')
    # Add xticks on the middle of the group bars
    plt.xlabel('Costs per dataset per LLM', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(costs[0]))], ['factool-qa', 'felm-wk', 'factcheckgpt'])
    plt.ylabel('Cost in USD', fontweight='bold')
    # Create legend & Show graphic
    plt.legend()
    plt.show()


###
# Main
###

if __name__ == "__main__":
    # Execute the factool pipeline solvers. Note - once the data is collected, this step can be commented out
    execute_factool_solvers()
    # Grouped barplot for false claims per dataset per LLM
    # gr_barplot_false_claims()
    # Grouped barplot for costs per dataset per LLM
    # gr_barplot_costs()
