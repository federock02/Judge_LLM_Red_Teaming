import gc
import warnings
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_persona import (
    get_initial_persona_messages,
    get_persona_mutation_messages,
    get_new_persona_messages,
    get_persona_refinement_messages,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Maximum number of tokens kept from parent/previous prompts before truncation
MAX_PROMPT_TOKENS = 512


class RedLLM:
    STOP_SEQUENCES = [
        "### Human:",
        "### Assistant:",
        "\nUser:",
        "</s>",
        "<|endoftext|>",
    ]

    # Max new tokens per generation type
    MAX_NEW_TOKENS = {
        "mutation":    512,
        "refinement":  512,
        "persona":     256,
    }

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

        self.temperature = 0.7
        self.top_p       = 0.9

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

    def _truncate_to_tokens(self, text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
        """Truncate text to at most max_tokens tokens (tail-preserving)."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[-max_tokens:])

    def _format_prompt(self, messages) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return messages

    @torch.inference_mode()
    def _generate_batch(self, prompts: List[str], generation_type: str) -> List[str]:
        """Batched generation with OOM fallback returning []."""
        max_new_tokens = self.MAX_NEW_TOKENS[generation_type]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        first_device = next(self.model.parameters()).device
        inputs       = {k: v.to(first_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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

    def _generate(self, messages, generation_type: str, num_prompts: int = 1) -> List[str]:
        """Generate with batch-size fallback on OOM."""
        formatted = self._format_prompt(messages)
        for batch_size in range(num_prompts, 0, -1):
            responses = self._generate_batch([formatted] * batch_size, generation_type)
            if responses:
                return responses
        return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate_initial_persona(self, parent_prompt: str) -> str:
        """Generate an initial persona JSON string for a given parent prompt."""
        messages = get_initial_persona_messages(parent_prompt)
        responses = self._generate(messages, generation_type="persona", num_prompts=1)
        raw = responses[0] if responses else ""
        if self.verbose:
            print(f"[ATTACKER] Initial persona:\n{raw}", flush=True)
        return raw

    def generate_mutated_prompts(
        self, parent_prompt: str, persona: dict, num_prompts: int
    ) -> List[str]:
        """First mutation: rewrite parent_prompt conditioned on persona."""
        messages  = get_persona_mutation_messages(parent_prompt, persona)
        responses = self._generate(messages, generation_type="mutation", num_prompts=num_prompts)
        if self.verbose:
            print(f"[ATTACKER] Mutated prompts:\n{responses}", flush=True)
        return responses

    def generate_refined_prompts(
        self, parent_prompt: str, previous_prompt: str, persona: dict, num_prompts: int
    ) -> List[str]:
        """Iterative refinement: improve previous_prompt while preserving persona."""
        messages = get_persona_refinement_messages(
            parent_prompt=parent_prompt,
            previous_prompt=self._truncate_to_tokens(previous_prompt),
            persona=persona,
        )
        responses = self._generate(messages, generation_type="refinement", num_prompts=num_prompts)
        if self.verbose:
            print(f"[ATTACKER] Refined prompts:\n{responses}", flush=True)
        return responses

    def generate_new_persona(
        self, parent_prompt: str, mutated_prompt: str, score: float, persona: dict
    ) -> str:
        """Generate an improved persona given a failed mutation and its judge score."""
        messages = get_new_persona_messages(
            parent_prompt=parent_prompt,
            mutated_prompt=self._truncate_to_tokens(mutated_prompt),
            score=score,
            persona=persona,
        )
        responses = self._generate(messages, generation_type="persona", num_prompts=1)
        raw = responses[0] if responses else ""
        if self.verbose:
            print(f"[ATTACKER] New persona:\n{raw}", flush=True)
        return raw