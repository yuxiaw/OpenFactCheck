import os
import time
from openai import OpenAI
import openai

client = None


def init_client():
    global client
    if client is None:
        if openai.api_key is None and 'OPENAI_API_KEY' not in os.environ:
            print("openai_key not presented, delay to initialize.")
            return
        client = OpenAI()


def request(
        user_inputs,
        model,
        system_role,
        temperature=1.0,
        return_all=False,
):
    init_client()

    if type(user_inputs) == str:
        chat_histories = [{"role": "user", "content": user_inputs}]
    elif type(user_inputs) == list:
        if all([type(x) == str for x in user_inputs]):
            chat_histories = [
                {
                    "role": "user" if i % 2 == 0 else "assistant", "content": x
                } for i, x in enumerate(user_inputs)
            ]
        elif all([type(x) == dict for x in user_inputs]):
            chat_histories = user_inputs
        else:
            raise ValueError("Invalid input for OpenAI API calling")
    else:
        raise ValueError("Invalid input for OpenAI API calling")
    

    messages = [{"role": "system", "content": system_role}] + chat_histories

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if return_all:
        return response
    response_str = ''
    for choice in response.choices:
        response_str += choice.message.content
    return response_str


def gpt(
        user_inputs,
        model,
        system_role,
        temperature=1.0,
        num_retries=3,
        waiting=1
):
    response = None
    for _ in range(num_retries):
        try:
            response = request(user_inputs, model, system_role, temperature=temperature)
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting)
    return response
