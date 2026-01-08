from datasets import load_dataset
from typing import List, Dict
import re


class UnsafeResponseDataset:
    """
    Loads unsafe LLM-generated responses from multiple datasets.
    Preserves original dataset IDs and adds source for disambiguation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.data: List[Dict] = []
        self.load_toxigen()
        self.load_realtoxicity_continuations()
        self.load_anthropic_assistant_turns()
        self.load_wildguard()
        self.load_aegis()

    # ============================================================
    # Public API
    # ============================================================

    def get_dataset(self) -> List[Dict]:
        """
        Returns a list of dicts with format:
        {
            "id": <original_id>,
            "source": <dataset_name>,
            "response": <text>
        }
        """
        return self.data

    # ============================================================
    # Dataset loaders
    # ============================================================

    def load_toxigen(self):
        idx = self.data.__len__()
        ds = load_dataset("toxigen/toxigen-data", split="train")

        if self.verbose:
            print(ds.column_names)

        for i, row in enumerate(ds):
            if row.get("toxicity_human", 0) > 0:
                text = row.get("text")
                if not text:
                    continue
                if len(text.split()) < 20:
                    continue
                self.data.append({
                    "id": row.get("id", str(i)),
                    "source": "toxigen",
                    "response": text
                })
        if self.verbose and len(self.data) > idx:
            print(f"Loaded {len(self.data)} unsafe responses from ToxiGen.", flush=True)
            print(f"Sample ID: {self.data[idx]['id']}, Response: {self.data[idx]['response']}", flush=True)

    def load_realtoxicity_continuations(self):
        idx = self.data.__len__()
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")

        if self.verbose:
            print(ds.column_names)

        for i, row in enumerate(ds):
            # use continuation only (LLM output)
            if row.get("continuation") is not None:
                cont = row.get("continuation")
                if isinstance(cont, dict):
                    text = cont.get("text", "")
                elif isinstance(cont, str):
                    text = cont
                else:
                    continue
                if len(text.split()) < 20:
                    continue
                self.data.append({
                    "id": row["prompt_id"] if "prompt_id" in row else f"{i}",
                    "source": "realtoxicity",
                    "response": text
                })
        
        if self.verbose and len(self.data) > idx:
            print(f"Loaded {len(self.data)} unsafe responses from RealToxicityPrompts.", flush=True)
            print(f"Sample ID: {self.data[idx]['id']}, Response: {self.data[idx]['response']}", flush=True)
    
    def extract_first_assistant_turn(self, text: str) -> str:
        """
        Extract the first assistant response from an Anthropic HH-RLHF dialogue.
        """
        pattern = r"Assistant:\s*(.*?)(?:\n\s*Human:|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def load_anthropic_assistant_turns(self):
        idx = self.data.__len__()
        ds = load_dataset("Anthropic/hh-rlhf", split="train")

        if self.verbose:
            print(ds.column_names)

        for i, row in enumerate(ds):
            raw = row.get("chosen", "")
            if not raw:
                continue

            assistant_text = self.extract_first_assistant_turn(raw)
            if len(assistant_text.split()) < 20:
                continue

            if assistant_text:
                self.data.append({
                    "id": row["id"] if "id" in row else f"{i}",
                    "source": "anthropic",
                    "response": assistant_text
                })
        
        if self.verbose and len(self.data) > idx:
            print(f"Loaded {len(self.data)} unsafe responses from Anthropic HH-RLHF.", flush=True)
            print(f"Sample ID: {self.data[idx]['id']}, Response: {self.data[idx]['response']}", flush=True)

    def load_wildguard(self):
        idx = self.data.__len__()
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", token=True)

        if self.verbose:
            print(ds.column_names)

        for i, row in enumerate(ds):
            if row.get("response_harm_label") != "harmful":
                continue
            if row.get("response_refusal_label") == "refusal":
                continue

            text = row.get("response")
            if not text:
                continue
            if len(text.split()) < 20:
                continue

            self.data.append({
                "id": row.get("id", str(i)),
                "source": "wildguard",
                "response": text
            })

        if self.verbose and len(self.data) > idx:
            print(f"Loaded {len(self.data) - idx} unsafe responses from WildGuardMix.", flush=True)
            print(f"Sample ID: {self.data[idx]['id']}, Response: {self.data[idx]['response'][:200]}...", flush=True)

    def load_aegis(self):
        idx = self.data.__len__()
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

        if self.verbose:
            print(ds.column_names)

        for i, row in enumerate(ds):
            if row.get("response_label") == "unsafe":
                if len(row["response"].split()) < 20:
                    continue
                self.data.append({
                    "id": row["id"] if "id" in row else f"{i}",
                    "source": "aegis",
                    "response": row["response"]
                })
        
        if self.verbose:
            print(f"Loaded {len(self.data)} unsafe responses from Aegis.", flush=True)
            print(f"Sample ID: {self.data[idx]['id']}, Response: {self.data[idx]['response']}", flush=True)
