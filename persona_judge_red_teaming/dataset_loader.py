import re
from typing import List, Dict
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
class UnsafeResponseDataset:
    """
    Loads unsafe LLM-generated responses from multiple datasets.
    Each entry has the form: {"id": str, "source": str, "response": str}
    """

    MIN_WORDS = 20

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.data: List[Dict] = []

        self._load_all()
        print(f"[DATASET LOADER] Total unsafe responses loaded: {len(self.data)}", flush=True)

    def get_dataset(self) -> List[Dict]:
        return self.data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        # --- Legacy loaders (kept for reference) ---
        # self._load(self._iter_toxigen,                  "toxigen")
        # self._load(self._iter_realtoxicity,             "realtoxicity")
        # self._load(self._iter_anthropic_assistant_turns,"anthropic")
        # self._load(self._iter_wildguard,                "wildguard")
        # self._load(self._iter_aegis,                    "aegis")
        # --- New loaders ---
        self._load(self._iter_beavertails,              "beavertails")
        self._load(self._iter_pku_saferlhf,             "pku_saferlhf")
        self._load(self._iter_harmbench,                "harmbench")
        self._load(self._iter_jbb_behaviors,            "jbb_behaviors")

    def _load(self, iterator_fn, source: str) -> None:
        """
        Calls iterator_fn(), appends yielded entries, and logs progress if verbose.
        iterator_fn must yield dicts with keys: id, response.
        """
        idx_before = len(self.data)
        for entry in iterator_fn():
            self.data.append({"source": source, **entry})
        n_added = len(self.data) - idx_before
        if self.verbose and n_added > 0:
            sample = self.data[idx_before]
            print(f"[DATASET LOADER] {source}: loaded {n_added} entries.", flush=True)
            print(f"[DATASET LOADER] {source} sample — ID: {sample['id']} | Response: {sample['response'][:200]}", flush=True)

    def _is_long_enough(self, text: str) -> bool:
        return len(text.split()) >= self.MIN_WORDS

    # ------------------------------------------------------------------
    # Dataset iterators
    # ------------------------------------------------------------------
    def _iter_toxigen(self):
        ds = load_dataset("toxigen/toxigen-data", split="train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            if not row.get("toxicity_human", 0):
                continue
            text = row.get("text", "")
            if not text or not self._is_long_enough(text):
                continue
            yield {"id": str(row.get("id", i)), "response": text}

    def _iter_realtoxicity(self):
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            cont = row.get("continuation")
            if cont is None:
                continue
            text = cont.get("text", "") if isinstance(cont, dict) else cont if isinstance(cont, str) else ""
            if not text or not self._is_long_enough(text):
                continue
            yield {"id": str(row.get("prompt_id", i)), "response": text}

    def _iter_anthropic_assistant_turns(self):
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            raw = row.get("chosen", "")
            if not raw:
                continue
            text = self._extract_first_assistant_turn(raw)
            if not text or not self._is_long_enough(text):
                continue
            yield {"id": str(row.get("id", i)), "response": text}

    def _iter_wildguard(self):
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", token=True)
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            if row.get("response_harm_label") != "harmful":
                continue
            if row.get("response_refusal_label") == "refusal":
                continue
            text = row.get("response", "")
            if not text or not self._is_long_enough(text):
                continue
            yield {"id": str(row.get("id", i)), "response": text}

    def _iter_aegis(self):
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            if row.get("response_label") != "unsafe":
                continue
            text = row.get("response", "")
            if not text or not self._is_long_enough(text):
                continue
            yield {"id": str(row.get("id", i)), "response": text}


    def _iter_beavertails(self):
        # Schema: prompt (str), response (str), category (dict[str, bool]), is_safe (bool)
        # We load the 330k split and filter to is_safe=False.
        # Each row maps a single LLM response to one or more harm categories;
        # we include the response text and store the active categories as metadata
        # in the id field for traceability.
        ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            if row.get("is_safe", True):
                continue
            text = row.get("response", "")
            if not text or not self._is_long_enough(text):
                continue
            # Collect active harm categories for the id
            cats = row.get("category", {})
            active = [k for k, v in cats.items() if v] if isinstance(cats, dict) else []
            yield {"id": f"bt_{i}", "response": text, "harm_categories": active}

    def _iter_pku_saferlhf(self):
        # PKU-SafeRLHF-QA flat per-response schema:
        # prompt (str), response (str), is_safe (bool),
        # harm_category (list[str]), severity_level (str: "minor"/"moderate"/"severe"/null)
        # The dataset is split across three subdirectories (one per source model).
        # We load all three and chain them to get the full 265k Q-A pairs.
        # Severity filter: keep "moderate" and "severe" only.
        from itertools import chain
        KEEP_SEVERITY = {2, 3}  # int: 0=safe, 1=minor, 2=moderate, 3=severe
        DATA_DIRS = ["data/Alpaca-7B", "data/Alpaca2-7B", "data/Alpaca3-8B"]
        rows = chain.from_iterable(
            load_dataset("PKU-Alignment/PKU-SafeRLHF-QA", data_dir=d, split="train")
            for d in DATA_DIRS
        )
        if self.verbose:
            print("[pku_saferlhf] loading 3 data dirs:", DATA_DIRS)
        severity_seen = set()
        for i, row in enumerate(rows):
            if row.get("is_safe", True):
                continue
            severity = row.get("severity_level")
            severity_seen.add(severity)
            # Accept both "moderate"/"severe" and numeric equivalents like "Level 2"/"Level 3"
            if severity not in KEEP_SEVERITY and severity not in {"Level 2", "Level 3"}:
                continue
            text = row.get("response", "")
            if not text or not self._is_long_enough(text):
                continue
            if i < 3 and self.verbose:
                print(f"[pku_saferlhf] severity_seen so far: {severity_seen}")
            cats = row.get("harm_category", {})
            active = [k for k, v in cats.items() if v] if isinstance(cats, dict) else []
            yield {"id": f"pku_{i}", "response": text, "severity": severity, "harm_categories": active}

    def _iter_harmbench(self):
        # Schema: Behavior (str), SemanticCategory (str), FunctionalCategory (str),
        #         Tags (list), ContextString (str, optional)
        # HarmBench contains only behavior descriptions (prompts), not LLM responses.
        # We use the Behavior field directly as the response text — these are
        # explicit harmful behavior descriptions, unambiguously unsafe, and
        # directly comparable to the red-teaming literature.
        # SemanticCategory and FunctionalCategory are stored for traceability.
        ds = load_dataset("walledai/HarmBench", "standard", split="train")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            text = row.get("Behavior", "")
            if not text or not self._is_long_enough(text):
                continue
            yield {
                "id":                  f"hb_{i}",
                "response":            text,
                "semantic_category":   row.get("SemanticCategory", ""),
                "functional_category": row.get("FunctionalCategory", ""),
            }

    def _iter_jbb_behaviors(self):
        # Schema: Behavior (str), Goal (str), Target (str), Category (str), Source (str)
        # JBB-Behaviors contains 100 curated misuse behaviors from AdvBench/HarmBench.
        # "Goal" is the harmful query string — more detailed than "Behavior".
        # "Target" is an affirmative response template — also useful as a seed.
        # We yield both Goal and Target as separate entries to maximize coverage,
        # since Target is an explicit harmful response and Goal is a harmful prompt.
        # MIN_WORDS filter is relaxed here since these are curated short behaviors.
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        if self.verbose:
            print(ds.column_names)
        for i, row in enumerate(ds):
            category = row.get("Category", "")
            # Yield Goal as a harmful prompt
            goal = row.get("Goal", "")
            if goal:
                yield {"id": f"jbb_goal_{i}", "response": goal, "category": category}
            # Yield Target as an explicit harmful response template
            target = row.get("Target", "")
            if target and self._is_long_enough(target):
                yield {"id": f"jbb_target_{i}", "response": target, "category": category}
    @staticmethod
    def _extract_first_assistant_turn(text: str) -> str:
        """Extract the first assistant response from an Anthropic HH-RLHF dialogue."""
        match = re.search(r"Assistant:\s*(.*?)(?:\n\s*Human:|\Z)", text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""