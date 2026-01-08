import torch
import warnings
import logging
import re
import argparse
import os
import random
import json
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_initial_persona_messages, get_persona_mutation_messages, get_new_persona_messages, get_persona_refinement_messages

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
        self.max_new_tokens_mutation = 512
        self.max_new_tokens_refinement = 512
        self.max_new_tokens_persona = 256
        self.stop_sequences = self._get_stop_sequences()

        print("[ATTACKER] Red LLM initialized.", flush=True)
        print("[ATTACKER] Model:", self.model_name, flush=True)
    
    
    def truncate(self, text, max_tokens=512):
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[-max_tokens:])
    
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
    def _generate_batch(self, prompts: List[str], type: str) -> List[str]:
        # Max new tokens based on generation type
        if type == "mutation":
            self.max_new_tokens = self.max_new_tokens_mutation
        elif type == "refinement":
            self.max_new_tokens = self.max_new_tokens_refinement
        elif type == "persona":
            self.max_new_tokens = self.max_new_tokens_persona
        else:
            raise ValueError(f"Unknown generation type: {type}")

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

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # remove prompt tokens: only keep the newly generated tokens
        new_tokens = [output[input_lengths:] for output in outputs]

        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        cleaned = [self._truncate_stop_sequences(text).strip() for text in decoded]
        return cleaned
    
    def _generate(self, prompt: str, type: str, num_prompts: int = 1) -> str:
        # Generate a single mutated prompt based on the input prompt
        formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False) if hasattr(self.tokenizer, "apply_chat_template") else prompt
        prompts = [formatted_prompt] * num_prompts
        responses = self._generate_batch(prompts, type=type)
        return responses

    def generate_initial_persona(self, parent_prompt: str):
        # Generate an initial persona based on the parent prompt
        messages = get_initial_persona_messages(parent_prompt)
        raw_output = self._generate(messages, type="persona", num_prompts=1)[0]
        return raw_output

    def generate_mutated_prompts(self, parent_prompt, persona, num_prompts):

        # Generate mutated prompts conditioned on a persona
        messages = get_persona_mutation_messages(parent_prompt, persona)
        responses = self._generate(messages, type="mutation", num_prompts=num_prompts)
        return responses
    
    def generate_refined_prompts(self, parent_prompt: str, previous_prompt: str, persona: dict, num_prompts: int):
        # Generate refined prompts based on previous prompt and persona
        messages = get_persona_refinement_messages(
            parent_prompt=parent_prompt,
            previous_prompt=self.truncate(previous_prompt),
            persona=persona
        )
        responses = self._generate(messages, type="refinement", num_prompts=num_prompts)
        return responses
    
    def generate_new_persona(self, parent_prompt: str, mutated_prompt: str, score: float, persona: dict):
        # Generate a new persona based on the performance of the mutated prompt
        # Return the raw output string as a specialized class will handle it
        messages = get_new_persona_messages(parent_prompt, self.truncate(mutated_prompt), score, persona)
        raw_output = self._generate(messages, type="persona", num_prompts=1)[0]

        return raw_output

    def refine_prompt(self, original_prompt: str, prompt: str, score: float, persona: dict, num_prompts: int) -> List[str]:
        # Refine a single prompt based on its score
        if self.verbose:
            print(f"\n[RED][DEBUG] Refining prompt with score {score}:\n{prompt}\n")
            
        # generate high-level advice
        new_persona = self.generate_new_persona(
            parent_prompt=original_prompt,
            mutated_prompt=prompt,
            score=score,
            persona=persona
        )

        # apply advice to generate new variants
        refined = self.generate_refined_prompts(
            parent_prompt=original_prompt,
            mutated_prompt=prompt,
            persona=new_persona,
            num_prompts=num_prompts,
        )
        
        """
        refined = self.generate_mutated_prompts(
            parent_prompt=original_prompt,
            persona=new_persona,
            num_prompts=num_prompts,
        )
        """

        return refined, new_persona
    
    def refine_prompts(self, original_prompt: str, prompts: List[str], scores: List[float], persona: dict, num_prompts: int) -> List[str]:
        # Refine a list of prompts based on their scores
        refined_prompts = []
        for prompt, score in zip(prompts, scores):
            if self.verbose:
                print(f"\n[RED][DEBUG] Refining prompt with score {score}:\n{prompt}\n")
            
            # generate high-level advice
            new_persona = self.generate_new_persona(
                parent_prompt=original_prompt,
                mutated_prompt=prompt,
                score=score,
                persona=persona
            )

            # apply advice to generate new variants
            refined = self.generate_refined_prompts(
                parent_prompt=original_prompt,
                mutated_prompt=prompt,
                persona=new_persona,
                num_prompts=num_prompts,
            )

            print(f"\n[RED][DEBUG] New Persona:\n{new_persona}\n")
            print(f"\n[RED][DEBUG] Refined Prompts:\n{refined}\n")
            
            """
            refined = self.generate_mutated_prompts(
                parent_prompt=original_prompt,
                persona=new_persona,
                num_prompts=num_prompts,
            )
            """

            refined_prompts.extend(refined)
        return refined_prompts, new_persona

    def __test__(self):
        risk_categories = get_risk_categories()
        attack_styles = get_attack_styles()

        r = random.choice(risk_categories)
        a = random.choice(attack_styles)
        print(f"\n[TESTING][RED] Selected Risk Category: {r}, Attack Style: {a}")

        # Example MUTATOR_PROMPT structure for testing
        initial_prompt = self.generate_initial_prompts(
            risk_category=r,
            attack_style=a,
            num_prompts=1
        )

        print("\n[TESTING][RED] Testing initial prompt generation.")
        print("\n[TESTING][RED] Initial Prompt Output:\n")
        print(initial_prompt[0])

        r = random.choice(risk_categories)
        a = random.choice(attack_styles)


        mutated_prompts = self.generate_mutated_prompts(
            parent_prompt=initial_prompt[0],
            risk_category=r,
            attack_style=a,
            num_prompts=5
        )

        print(f"\n[TESTING][RED] Testing batch generation ({len(mutated_prompts)} mutations):")
        print(f"\n[TESTING][RED] Risk Category: {r}, Attack Style: {a}")
        for i, response in enumerate(mutated_prompts):
            print(f"\n[TESTING][RED] Batch Item {i+1} Mutated Prompt:")
            print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private models")
    args = parser.parse_args()
    hf_token = args.hf_token
    if hf_token is not None:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    red = RedLLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", verbose=True)
    red.__test__()


