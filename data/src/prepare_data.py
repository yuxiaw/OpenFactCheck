# https://platform.openai.com/docs/guides/text-generation/json-mode
import time
import openai
from openai import OpenAI
client = OpenAI(api_key = "openaikey")  # put your openaikey 
import pandas as pd

IDENTIFY_DOMAIN_TOPIC_PROMPT = """
Given a (question, answer) pair, identify the domain and topic of the content. 
Your task is to output a JSON dictionary with 'domain' representing the general field of knowledge and 'topic' representing the specific subject within that domain.

Domains can include, but are not limited to, fields such as science, technology, literature, history, and more.
Topics are more specific and represent the subject matter discussed in the answer.

For example,
Question: "What are the main causes of climate change?"
Answer: "Climate change is primarily caused by human activities such as burning fossil fuels, deforestation, and industrial processes."

You should output a python dict without any other words.
dict("domain": "Environmental Science", "topic": "Causes of Climate Change")


Question: {question}
Answer: {response}
"""

def save_to_file(text, filename='error_output.txt'):
    """Save a string to a file line by line."""
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text + '\n')

def gpt_json_mode(user_input, model="gpt-3.5-turbo-1106",
                       system_role = "You are a helpful assistant designed to output JSON.",
                       num_retries=3, waiting_time = 1):
    r = ''
    for _ in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={ "type": "json_object"},
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_input}
                ])
            r = response.choices[0].message.content
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting_time)
    return r
    

def identify_domain_and_topic(datadir="factuality_eval_llms.jsonl", 
                              savedir = "domain_topic.jsonl", 
                              model="gpt-3.5-turbo-1106",
                              system_role = "You are a helpful assistant designed to output JSON, good at identifying text domain and topics.",
                              num_retries=3):
    # read data
    df = pd.read_json(datadir, lines=True)
    print(df.columns)
    # df.cat.value_counts()
    # We know the domain and topic for the first 1500 examples from snowballing
    df = df[1500:]
    print(len(df))

    # Test
    # df = df[:2]
    responses = [{"domain": "Mathematics", "topic": "Primality Testing"}] * 500 + \
    [{"domain": "History", "topic": "US Senator Search"}] * 500 + \
    [{"domain": "Transportation", "topic": "Graph Connectivity-Flight Search"}] * 500

    for i, row in df.iterrows():
        r = {}
        user_input = IDENTIFY_DOMAIN_TOPIC_PROMPT.format(question=row['question'], response=row['GPT4_response'])
        for _ in range(num_retries):
            try:
                r = gpt_json_mode(user_input, model=model, system_role=system_role)
                r = eval(r) # a json dict
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}.")
                save_to_file(r)
        responses.append(r)
        if i % 10 == 0:
            pd.DataFrame(responses).to_json(savedir, lines=True, orient="records", force_ascii=False)
    pd.DataFrame(responses).to_json(savedir, lines=True, orient="records", force_ascii=False)
    return responses