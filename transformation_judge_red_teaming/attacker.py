import gc
import warnings
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    get_attacker_mutate_template,
    get_attacker_guided_template,
    get_attacker_refinement_advice_template,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Maximum token budget for parent prompts (words * 1.3 ≈ tokens)
MAX_PARENT_WORDS = 1024


class RedLLM:
    STOP_SEQUENCES = [
        "### Human:",
        "### Assistant:",
        "\nUser:",
        "</s>",
        "<|endoftext|>",
    ]

    def __init__(self, model_name: str, device: str = "cuda", verbose: bool = False):
        self.model_name = model_name
        self.device     = device
        self.verbose    = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature    = 0.7
        self.top_p          = 0.9
        self.max_new_tokens = 1024

        print(f"[ATTACKER] Initialized {self.model_name} on {self.device}", flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _truncate_at_stop_sequence(self, text: str) -> str:
        for seq in self.STOP_SEQUENCES:
            idx = text.find(seq)
            if idx != -1:
                text = text[:idx]
        return text

    def _truncate_parent_prompt(self, prompt: str) -> str:
        """Truncate to MAX_PARENT_WORDS to stay within context budget."""
        words = prompt.split()
        return " ".join(words[:MAX_PARENT_WORDS]) if len(words) > MAX_PARENT_WORDS else prompt

    def _format_prompt(self, messages) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return messages

    @torch.inference_mode()
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Run batched generation; returns [] on OOM."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        first_device = next(self.model.parameters()).device
        inputs       = {k: v.to(first_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

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
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                print(f"[ATTACKER] OOM on batch size {len(prompts)}, clearing cache.", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                return []
            raise

        new_tokens = [out[input_length:] for out in outputs]
        decoded    = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        return [self._truncate_at_stop_sequence(t).strip() for t in decoded]

    def _generate(self, prompt, num_prompts: int) -> List[str]:
        """
        Attempt generation with batch size num_prompts, falling back to
        smaller batches on OOM until batch size 1.
        """
        formatted = self._format_prompt(prompt)
        for batch_size in range(num_prompts, 0, -1):
            responses = self._generate_batch([formatted] * batch_size)
            if responses:
                return responses
        return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate_mutated_prompts(
        self, parent_prompt: str, transformation: dict, num_prompts: int
    ) -> List[str]:
        parent_prompt = self._truncate_parent_prompt(parent_prompt)
        prompt = get_attacker_mutate_template(
            parent_prompt=parent_prompt,
            transformation=transformation["name"],
            description=transformation["description"],
        )
        responses = self._generate(prompt, num_prompts)
        if self.verbose:
            print(f"[ATTACKER] Mutated prompts:\n{responses}", flush=True)
        return responses

    def generate_guided_prompts(
        self,
        parent_prompt: str,
        mutated_prompt: str,
        transformation: dict,
        advice: str,
        num_prompts: int,
    ) -> List[str]:
        prompt = get_attacker_guided_template(
            parent_prompt=parent_prompt,
            mutated_prompt=mutated_prompt,
            transformation=transformation["name"],
            description=transformation["description"],
            advice=advice,
        )
        responses = self._generate(prompt, num_prompts)
        if self.verbose:
            print(f"[ATTACKER] Guided prompts:\n{responses}", flush=True)
        return responses

    def generate_refinement_advice(
        self, mutated_prompt: str, parent_prompt: str, score: float, transformation: dict
    ) -> str:
        prompt = get_attacker_refinement_advice_template(
            parent_prompt=parent_prompt,
            mutated_prompt=mutated_prompt,
            transformation=transformation["name"],
            description=transformation["description"],
            score=score,
        )
        responses = self._generate(prompt, num_prompts=1)
        advice = responses[0] if responses else ""
        if self.verbose:
            print(f"[ATTACKER] Refinement advice:\n{advice}", flush=True)
        return advice

    def refine_prompts(
        self,
        prompts: List[str],
        scores: List[float],
        original_prompt: str,
        transformation: dict,
        num_prompts: int,
        test: bool = False,
    ) -> List[str]:
        original_prompt = self._truncate_parent_prompt(original_prompt)
        refined = []
        for prompt, score in zip(prompts, scores):
            if self.verbose:
                print(f"[ATTACKER] Refining prompt (score={score:.4f}):\n{prompt}", flush=True)

            advice = self.generate_refinement_advice(
                mutated_prompt=prompt,
                parent_prompt=original_prompt,
                score=score,
                transformation=transformation,
            )
            if test:
                print(f"[ATTACKER] Refinement advice:\n{advice}", flush=True)

            refined.extend(self.generate_guided_prompts(
                parent_prompt=original_prompt,
                mutated_prompt=prompt,
                transformation=transformation,
                advice=advice,
                num_prompts=num_prompts,
            ))
        return refined