# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

import vllm
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

    
class LLaMA3(LM):
    def __init__(self, model_name, cache_file=None, tensor_parallel_size=2):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size  # the number of GPU cards
        if cache_file:
            super().__init__(cache_file)


    def load_model(self):
        self.model = vllm.LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def _generate(self, prompt, max_output_length=256, max_sequence_length=2048, 
                  system_prompt = "You are a helpful assistant."):
        """samples: list[str], a list of prompts"""
        samples = prompt
        is_single = isinstance(samples, str)
        if is_single:
            samples = [samples]

        if self.model is None:
            self.load_model()

        # print(tokenizer.chat_template)
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

        result = self.model.generate(model_inputs, sampling_params)
        responses = [ x.outputs[0].text.strip() for x in result]
        responses = [" ".join(x.split('\n\n')[1:]) for x in responses]

        assert len(responses)==len(samples)
        if is_single:
            return responses[0], result[0]
        
        return responses



# LLaMA1-Instruct
class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

