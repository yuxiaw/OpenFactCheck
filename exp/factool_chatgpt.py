import asyncio
import pandas as pd
import os
import sys
import time
import json
import math
from typing import List, Dict, Any

factool_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'factool-main', 'factool')
sys.path.insert(0, factool_dir)

from factool import Factool

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
os.environ['SERPER_API_KEY'] = 'SERPER_API_KEY'
# os.environ['SCRAPER_API_KEY'] = 'SCRAPER_API_KEY'
gpt_model = "gpt-3.5-turbo-0125"

def read_claim_data(filename='Factbench.jsonl', projectdir='./'):
    path = os.path.join(projectdir, filename)
    df = pd.read_json(path, lines=True)
    # Filter only the ones with source=factool-qa, source=felm-wk
    df = df[df["source"].str.contains('factool-qa|felm-wk')]
    # Extract data
    data = {
        "source": list(df["source"]),
        "prompt": list(df["prompt"]),
        "response": list(df["response"]),
        "claims": list(df["claims"]),
        "claim_labels": list(df["claim_labels"])
    }
    print("Items: ", len(data["source"]))
    
    return data

# Extract number and type of claims => that is, Exact Matching (EM)
def assess_experiment(claim_info):
    for k, v in claim_info.items():
        # If stance contains definitive or mixed, then it is false
        if "definitive" in v["stances"][0] or "mixed" in v["stances"][0]: # Mixed
            v["label"] = 2
        elif "error" in v["stances"][0] or "incorrect" in v["stances"][0] or "false" in v["stances"][0] or "not factual" in v["stances"][0] or "non-factual" in v["stances"][0] or "not accurate" in v["stances"][0] or "not entirely accurate" in v["stances"][0]: # False
            v["label"] = 0
        elif "factual" in v["stances"][0] or "confirm" in v["stances"][0] or "true" in v["stances"][0]or "accurate" in v["stances"][0]: # True
            v["label"] = 1
        else: # Unknown
            v["label"] = -1
            print (k, ':', v['stances'][0], '\n')

def evidence_processor(claims: list):
    factool_instance = Factool(gpt_model)
    # Evidence retrieval
    claims_in_response = [{ 'claim': claim } for claim in claims]
    queries = asyncio.run(factool_instance.pipelines["kbqa_online"]._query_generation(claims_in_response))
    search_outputs_for_claims = asyncio.run(factool_instance.pipelines["kbqa_online"].tool.run(queries))
    claim_info: Dict[str, Dict[str, Any]] = {}
    for i, claim in enumerate(claims):
        evidence_list: List[dict] = []
        #print(f'SOFC => {search_outputs_for_claims[i]}')
        if i >= len(search_outputs_for_claims):
            print(f'SOFC => search_outputs_for_claims[{i}] out of range - {i}. Bypassing missing evidence...')
            #continue
        else:
            for j, search_outputs_for_claim in enumerate(search_outputs_for_claims[i]):
                evidence_list.append({"evidence_id": j,
                                      "web_page_snippet_manual": search_outputs_for_claim['content'],
                                      "query": [queries[i]],
                                      "url": search_outputs_for_claim['source'],
                                      "web_text": []})
        claim_info[claim] = { "claim": claim, "automatic_queries": queries[i], "evidence_list": evidence_list }
    # Claim examiner
    evidences = [[output['content'] for output in search_outputs_for_claim] for search_outputs_for_claim in search_outputs_for_claims]
    verifications = asyncio.run(factool_instance.pipelines["kbqa_online"]._verification(claims_in_response, evidences))
    for index, (key, pair) in enumerate(claim_info.items()):
        stance = ''
        if (type(verifications[index]) == None or verifications[index] == 'None' or verifications[index] == None):
            stance = claims_in_response[index]['claim']
        else:
            stance = ('' if (verifications[index]['error'] == None or verifications[index]['error'] == 'None' or len(verifications[index]['error']) == 0) else (
                        verifications[index]['error'] + ' '))
            stance += (
                '' if (verifications[index]['reasoning'] == None or verifications[index]['reasoning'] == 'None' or len(verifications[index]['reasoning']) == 0) else
                verifications[index]['reasoning'])
            stance += (claims_in_response[index]['claim'] if (
                        verifications[index]['correction'] == None or verifications[index]['correction'] == 'None' or len(
                    verifications[index]['correction']) == 0) else (' ' + verifications[index]['correction']))
        claim_info[key]['stances'] = [stance]
        for j in range(len(claim_info[key]['evidence_list'])):
            claim_info[key]['evidence_list'][j]['stance'] = stance
    
    return claim_info

def execute_factool_solvers():
    response_data = read_claim_data()
    for i in range(0, len(response_data["prompt"])):
        prompt = response_data["prompt"][i]
        claims = response_data["claims"][i]
        claim_labels = response_data["claim_labels"][i]
        response = response_data["response"][i]
        start = time.time() * 1000
        claim_info = evidence_processor(claims)
        for index, (k, v) in enumerate(claim_info.items()):
            v["prior_label"] = claim_labels[index]
        end = time.time() * 1000
        # Process the result data
        assess_experiment(claim_info)
        # Collect in data object
        claim_data = {}
        claim_data["claims_info"] = claim_info
        claim_data["source"] = response_data["source"][i]
        claim_data["prompt"] = prompt
        claim_data["response"] = response
        claim_data["start"] = math.floor(start)
        claim_data["end"] = math.floor(end)
        # Keep the results
        print ("Done with prompt [", i, "] '", prompt,"' in ", end - start, " ms.")
        # Save to jsonl file
        with open("FactoolBench.jsonl", "a") as outfile:
            json.dump(claim_data, outfile)
            outfile.write('\n')

# Main function
if __name__ == "__main__":
    execute_factool_solvers()