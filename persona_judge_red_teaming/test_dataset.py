"""
test_dataset_loader.py — Samples a few entries from each dataset iterator
to verify schemas, filters, and content quality without loading everything.

Usage:
    python test_dataset_loader.py [--hf_token hf_...]
"""
from __future__ import annotations

import argparse
import os
from datasets import load_dataset

SAMPLES_PER_DATASET = 3
MIN_WORDS = 20


def _is_long_enough(text: str) -> bool:
    return len(text.split()) >= MIN_WORDS


def sample_dataset(name: str, iterator_fn, n: int = SAMPLES_PER_DATASET) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    collected = 0
    try:
        for entry in iterator_fn():
            print(f"\n  [{collected+1}] id={entry.get('id')}")
            print(f"      response : {entry.get('response', '')[:300]}")
            # print any extra metadata fields
            for k, v in entry.items():
                if k not in ("id", "response"):
                    print(f"      {k:20s}: {v}")
            collected += 1
            if collected >= n:
                break
        if collected == 0:
            print("  [WARNING] No entries passed the filter.")
    except Exception as e:
        print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Minimal inline iterators (mirrors dataset_loader.py, no class overhead)
# ---------------------------------------------------------------------------

def iter_toxigen():
    ds = load_dataset("toxigen/toxigen-data", split="train")
    for i, row in enumerate(ds):
        if not row.get("toxicity_human", 0):
            continue
        text = row.get("text", "")
        if not text or not _is_long_enough(text):
            continue
        yield {"id": str(row.get("id", i)), "response": text}

def iter_realtoxicity():
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    for i, row in enumerate(ds):
        cont = row.get("continuation")
        if cont is None:
            continue
        text = cont.get("text", "") if isinstance(cont, dict) else cont if isinstance(cont, str) else ""
        if not text or not _is_long_enough(text):
            continue
        yield {"id": str(row.get("prompt_id", i)), "response": text}

def iter_anthropic():
    import re
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    for i, row in enumerate(ds):
        raw = row.get("chosen", "")
        if not raw:
            continue
        match = re.search(r"Assistant:\s*(.*?)(?:\n\s*Human:|\Z)", raw, flags=re.DOTALL)
        text = match.group(1).strip() if match else ""
        if not text or not _is_long_enough(text):
            continue
        yield {"id": str(row.get("id", i)), "response": text}

def iter_wildguard():
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", token=True)
    for i, row in enumerate(ds):
        if row.get("response_harm_label") != "harmful":
            continue
        if row.get("response_refusal_label") == "refusal":
            continue
        text = row.get("response", "")
        if not text or not _is_long_enough(text):
            continue
        yield {"id": str(row.get("id", i)), "response": text}

def iter_aegis():
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    for i, row in enumerate(ds):
        if row.get("response_label") != "unsafe":
            continue
        text = row.get("response", "")
        if not text or not _is_long_enough(text):
            continue
        yield {"id": str(row.get("id", i)), "response": text}

def iter_beavertails():
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    for i, row in enumerate(ds):
        if row.get("is_safe", True):
            continue
        text = row.get("response", "")
        if not text or not _is_long_enough(text):
            continue
        cats = row.get("category", {})
        active = [k for k, v in cats.items() if v] if isinstance(cats, dict) else []
        yield {"id": f"bt_{i}", "response": text, "harm_categories": active}

def iter_pku_saferlhf():
    from itertools import chain
    KEEP_SEVERITY = {2, 3}  # int: 0=safe, 1=minor, 2=moderate, 3=severe
    DATA_DIRS = ["data/Alpaca-7B", "data/Alpaca2-7B", "data/Alpaca3-8B"]
    rows = chain.from_iterable(
        load_dataset("PKU-Alignment/PKU-SafeRLHF-QA", data_dir=d, split="train")
        for d in DATA_DIRS
    )
    for i, row in enumerate(rows):
        if row.get("is_safe", True):
            continue
        severity = row.get("severity_level")
        if severity not in KEEP_SEVERITY:
            continue
        text = row.get("response", "")
        if not text or not _is_long_enough(text):
            continue
        yield {"id": f"pku_{i}", "response": text, "severity": severity}

def iter_harmbench():
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    for i, row in enumerate(ds):
        text = row.get("prompt", "")
        if not text:
            continue  # no MIN_WORDS filter — HarmBench prompts are intentionally short
        yield {
            "id":       f"hb_{i}",
            "response": text,
            "category": row.get("category", ""),
        }

def iter_jbb_behaviors():
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    for i, row in enumerate(ds):
        category = row.get("Category", "")
        goal = row.get("Goal", "")
        if goal:
            yield {"id": f"jbb_goal_{i}", "response": goal, "category": category}
        target = row.get("Target", "")
        if target and _is_long_enough(target):
            yield {"id": f"jbb_target_{i}", "response": target, "category": category}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DATASETS = [
    ("toxigen",       iter_toxigen),
    ("realtoxicity",  iter_realtoxicity),
    ("anthropic",     iter_anthropic),
    ("wildguard",     iter_wildguard),
    ("aegis",         iter_aegis),
    ("beavertails",   iter_beavertails),
    ("pku_saferlhf",  iter_pku_saferlhf),
    ("harmbench",     iter_harmbench),
    ("jbb_behaviors", iter_jbb_behaviors),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of dataset names to test, e.g. beavertails,harmbench")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    targets = (
        {d.strip() for d in args.only.split(",")}
        if args.only else
        {name for name, _ in DATASETS}
    )

    for name, fn in DATASETS:
        if name in targets:
            sample_dataset(name, fn)

    print("\n[DONE]")