import vllm
import torch
from transformers import AutoTokenizer

verifier_prompt = """
You are given a piece of text. Your task is to identify whether there are any factual errors within the text.
When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences may be helpful. Some evidences may contradict to each other. You must be careful when using the evidences to judge the factuality of the given text.
    
The response should be a dictionary with three keys - "reasoning", "factuality", "error", and "correction", which correspond to the reasoning, whether the given text is factual or not (Boolean - True or False), the factual error present in the text, and the corrected text.
    
The following is the given text
[text]: {claim}
The following is the provided evidences
[evidences]: {evidence}

You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '{{'.
[response format]: 
{{
    "reasoning": "Why is the given text factual or non-factual? Be careful when you said something is non-factual. When you said something is non-factual, you must provide multiple evidences to support your decision.",
    "error": "None if the text is factual; otherwise, describe the error.",
    "correction": "The corrected text if there is an error.",
    "factuality": True if the given text is factual, False otherwise.
}}
"""

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