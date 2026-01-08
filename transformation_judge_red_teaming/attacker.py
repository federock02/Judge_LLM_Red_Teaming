import torch
import warnings
import logging
import re
import argparse
import os
import random
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import  get_attacker_mutate_template, get_transformations, get_attacker_guided_template, get_attacker_refinement_advice_template

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class RedLLM:
    def __init__(self, model_name: str, device: str = "cuda", verbose: bool = False):
        self.verbose = verbose
        self.model_name = model_name
        self.device = device

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            #device_map="auto",
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Sampling parameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_new_tokens = 1024
        self.stop_sequences = self._get_stop_sequences()

        print("[ATTACKER] Red LLM initialized.", flush=True)
        print("[ATTACKER] Model:", self.model_name, flush=True)
    
    def _get_stop_sequences(self) -> list:
        # Returns a list of default strings that will be used to stop generation.
        # These are generic markers to prevent the mutator from continuing the dialogue.
        return [
            "### Human:",
            "### Assistant:",
            "\nUser:",
            "</s>", 
            "<|endoftext|>"
        ]

    def _truncate_stop_sequences(self, text: str) -> str:
        # Stop generation at the first occurrence of any stop sequence.
        for seq in self.stop_sequences:
            idx = text.find(seq)
            if idx != -1:
                text = text[:idx]
        return text
    
    @torch.inference_mode()    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # Batched generation
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # move inputs to model device
        # model may be sharded with device_map="auto", so use device of first param
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        input_lengths = inputs["input_ids"].shape[1]  # keep track of prompt length

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"[ATTACKER][ERROR] CUDA OOM during generation (batch={len(prompts)}). Clearing cache and returning empty.", flush=True)
            torch.cuda.empty_cache()
            import gc; gc.collect()
            return []
        except RuntimeError as e:
            # Catch generic runtime errors; check if OOM-related
            if 'out of memory' in str(e).lower():
                print(f"[ATTACKER][ERROR] CUDA OOM (RuntimeError) during generation (batch={len(prompts)}). Clearing cache.", flush=True)
                torch.cuda.empty_cache()
                import gc; gc.collect()
                return []
            else:
                # Re-raise other runtime errors (don't silently swallow them)
                raise


        # remove prompt tokens: only keep the newly generated tokens
        new_tokens = [output[input_lengths:] for output in outputs]

        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        cleaned = [self._truncate_stop_sequences(text).strip() for text in decoded]
        return cleaned
    
    def _generate(self, prompt: str, num_prompts: int = 1) -> str:
        # Generate a single mutated prompt based on the input prompt
        formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False) if hasattr(self.tokenizer, "apply_chat_template") else prompt
        for i in range(num_prompts, 0, -1):
            prompts = [formatted_prompt] * i
            responses = self._generate_batch(prompts)
            if responses != []:
                return responses
        return responses
    
    def check_parent_length(self, parent_prompt: str) -> str:
        # estimate the number of tokens in the parent prompt, 1 word ~= 1.3 tokens
        word_count = len(parent_prompt.split())
        estimated_tokens = int(word_count * 1.3)
        if estimated_tokens > 1024:
            return " ".join(parent_prompt.split()[:1024])  # truncate to first 1024 words
        return parent_prompt

    def generate_mutated_prompts(self, parent_prompt: str, transformation: dict, num_prompts: int) -> str:
        # Generate a mutated prompt based on the parent prompt
        parent_prompt = self.check_parent_length(parent_prompt)
        prompt = get_attacker_mutate_template(parent_prompt=parent_prompt, transformation=transformation["name"], description=transformation["description"])

        responses = self._generate(prompt=prompt, num_prompts=num_prompts)
        if self.verbose:
            print(f"\n[RED][DEBUG] Generated mutated prompts:\n{responses}\n")
        return responses
    
    def generate_guided_prompts(self, parent_prompt: str, mutated_prompt: str, transformation: dict, advice: str, num_prompts: int) -> str:
        # Generate a mutated prompt based on the parent prompt and advice
        prompt = get_attacker_guided_template(
            parent_prompt=parent_prompt,
            mutated_prompt=mutated_prompt,
            transformation=transformation["name"],
            description=transformation["description"],
            advice=advice,
        )

        responses = self._generate(prompt=prompt, num_prompts=num_prompts)
        if self.verbose:
            print(f"\n[RED][DEBUG] Generated guided prompts:\n{responses}\n")
        return responses
    
    def generate_refinement_advice(self, mutated_prompt: str, parent_prompt: str, score: float, transformation: dict) -> str:
        # Generate advice for refining a prompt based on its score
        prompt = get_attacker_refinement_advice_template(
            parent_prompt=parent_prompt,
            mutated_prompt=mutated_prompt,
            transformation=transformation["name"],
            description=transformation["description"],
            score=score,
        )

        responses = self._generate(prompt=prompt, num_prompts=1)
        if self.verbose:
            print(f"\n[RED][DEBUG] Generated refinement advice:\n{responses[0]}\n")
        return responses[0]
    
    def refine_prompts(self, prompts: List[str], scores: List[float], original_prompt: str, transformation: dict, num_prompts: int, test: bool=False) -> List[str]:
        # Refine a list of prompts based on their scores
        refined_prompts = []
        original_prompt = self.check_parent_length(original_prompt)
        for prompt, score in zip(prompts, scores):
            if self.verbose:
                print(f"\n[RED][DEBUG] Refining prompt with score {score}:\n{prompt}\n")
            
            # generate high-level advice
            advice = self.generate_refinement_advice(
                mutated_prompt=prompt,
                parent_prompt=original_prompt,
                score=score,
                transformation=transformation
            )
            if test:
                print(f"REFINEMENT ADVICE:\n{advice}\n")

            # apply advice to generate new variants
            refined = self.generate_guided_prompts(
                parent_prompt=original_prompt,
                mutated_prompt=prompt,
                transformation=transformation,
                advice=advice,
                num_prompts=num_prompts,
            )

            refined_prompts.extend(refined)
        return refined_prompts


