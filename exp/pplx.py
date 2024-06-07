import time
import pandas as pd
import numpy as np
from openai import OpenAI

def _call_pplxai(prompt, model_name = "sonar-small-online", key_path="pplx_apikey.txt", 
                 system_prompt = "You are an artificial intelligence assistant and you need to verify the factuality of some claims."):
    with open(key_path, 'r') as f:
        API_KEY = f.readline().strip()
    # print(API_KEY)
    client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
            ),
        },
        {
            "role": "user",
            "content": (
                prompt
            ),
        },
    ]
    
    # chat completion without streaming
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result
    
def call_pplxai(user_input, model="sonar-small-online", key_path = "pplx_apikey.txt", 
                system_prompt="You are a helpful assistant.", 
                num_retries=3, waiting_time=1):
    r = ''
    for _ in range(num_retries):
        try:
            r = _call_pplxai(prompt=user_input, model_name=model, key_path=key_path, system_prompt=system_prompt)
            break
        except:
            print("Retrying...")
            time.sleep(np.power(2, _))
    return r