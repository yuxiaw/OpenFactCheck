# author: yuxiaw
# time: 04/27/2024

# ----------------------------------------------------------------------
# OpenAI GPT
# ----------------------------------------------------------------------
# version update
# https://github.com/openai/openai-python/discussions/742
import openai
import time
# key_path = "../data/openaikey.txt"
# with open(key_path, 'r') as f:
#     api_key = f.readline()
# openai.api_key = api_key.strip()
openai.api_key = "openai_api_key"

def gpt_single_easy_try(user_input, model="gpt-3.5-turbo",
                        system_role = "You are a helpful assistant."):
    response = openai.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_input},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def gpt_easy(user_input, model="gpt-3.5-turbo", 
             system_role="You are a helpful assistant.", num_retries=3, waiting_time=2):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_easy_try(user_input, model=model, system_role=system_role)
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting_time)
    return r


# ----------------------------------------------------------------------
# LLaMA3
# ----------------------------------------------------------------------
import vllm
import torch
from transformers import AutoTokenizer

class LLaMA3(object):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size  # the number of GPU cards
        self.model = None
        

    def load_model(self):
        # load the model and put it as self.model
        self.model = vllm.LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # print(tokenizer.chat_template)


    def generate(self, samples, max_output_length=256, system_prompt = "You are a helpful assistant."):
        """samples: list[str], a list of prompts"""
        if isinstance(samples, str):
            samples = [samples]

        if self.model is None:
            self.load_model()

        # set generation parameters. this can be configurated
        sampling_params = vllm.SamplingParams(max_tokens=max_output_length)
        # print(sampling_params)
        model_inputs = [
            self.tokenizer.apply_chat_template(
                [
                    {"role":"system", "content": system_prompt.strip()}, 
                    {"role":"user", "content": sample.strip()}
                ], 
                tokenize=False)
            for sample in samples
        ]
        # print(model_inputs[0])
        
        result = self.model.generate(model_inputs, sampling_params) 
        # outputs[0] is get the top if sampling several.
        responses = [x.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip() for x in result]

        return responses