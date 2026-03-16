"""
metrics.py — shared data structures and metric functions for both
transformation-based and persona-based red-teaming analysis.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationEdge:
    # identity
    iteration_id:    int
    refinement_iter: int          # 0 = unmutated root, 1 = first mutation, ...

    # root context
    root_prompt:     str
    root_fitness:    float
    source_dataset:  Optional[str]

    # mutation context
    parent_prompt:   str
    parent_fitness:  float
    child_prompt:    str
    child_fitness:   float
    delta_fitness:   float
    success:         bool

    # operator metadata
    operator_type:     str        # "transformation" or "persona"
    operator_name:     str        # transformation name or cluster id
    operator_metadata: Dict[str, Any]

    # convenience flag
    is_root_edge: bool            # True iff parent_prompt == root_prompt


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self.cache[text]
    

    def batch_embed(self, texts: list, batch_size: int = 256) -> None:
        """
        Encode all texts not yet in cache in a single batched call.
        Much faster than calling embed() in a loop when dealing with
        thousands of unique strings, since sentence-transformers
        parallelises encoding across the batch internally.
        Call this once before any hot loop that calls cosine().
        """
        unseen = [t for t in texts if t not in self.cache]
        if not unseen:
            return
        print(f"[EMBEDDER] Batch encoding {len(unseen)} unique texts...", flush=True)
        vecs = self.model.encode(
            unseen,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for text, vec in zip(unseen, vecs):
            self.cache[text] = vec

    def cosine(self, a: str, b: str) -> float:
        return float(np.dot(self.embed(a), self.embed(b)))


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_edges_by_operator(edges: List[MutationEdge]) -> Dict[str, List[MutationEdge]]:
    grouped: Dict[str, List[MutationEdge]] = defaultdict(list)
    for e in edges:
        grouped[e.operator_name].append(e)
    return dict(grouped)


def group_edges_by_root(edges: List[MutationEdge]) -> Dict[str, List[MutationEdge]]:
    by_root: Dict[str, List[MutationEdge]] = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)
    return dict(by_root)


# ---------------------------------------------------------------------------
# Attack Success Rate (ASR)
# ---------------------------------------------------------------------------

def attack_success_rate(edges: List[MutationEdge]) -> float:
    """Edge-level ASR: fraction of all mutation edges that succeeded."""
    if not edges:
        return 0.0
    return float(np.mean([e.success for e in edges]))


def root_level_attack_success_rate(edges: List[MutationEdge]) -> float:
    """Root-level ASR: fraction of root prompts for which at least one mutation succeeded."""
    if not edges:
        return 0.0
    by_root = group_edges_by_root(edges)
    successes = sum(1 for es in by_root.values() if any(e.success for e in es))
    return successes / len(by_root)


def attack_success_rate_per_operator(
    grouped: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {name: attack_success_rate(es) for name, es in grouped.items()}


def root_level_asr_per_operator(
    grouped: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {name: root_level_attack_success_rate(es) for name, es in grouped.items()}


# ---------------------------------------------------------------------------
# Semantic Preservation (SP)
# ---------------------------------------------------------------------------

def semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> List[float]:
    """Per-edge cosine similarity between child and reference (root or parent)."""
    ref = lambda e: e.root_prompt if reference == "root" else e.parent_prompt
    return [embedder.cosine(ref(e), e.child_prompt) for e in edges]


def semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> Dict[str, float]:
    vals = semantic_preservation(edges, embedder, reference=reference)
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std":  float(np.std(vals))  if vals else 0.0,
    }


def root_level_semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> List[float]:
    """
    Per-root SP: for each root, consider terminal edges (successful OR at max depth),
    compute cosine similarity of each to the reference, and average within the root.
    Returns one value per unique root prompt.
    """
    if not edges:
        return []
    by_root = group_edges_by_root(edges)
    values = []
    for root, es in by_root.items():
        max_depth = max(e.refinement_iter for e in es)
        terminals = [e for e in es if e.success or e.refinement_iter == max_depth]
        ref = lambda e: e.root_prompt if reference == "root" else e.parent_prompt
        sims = [embedder.cosine(ref(e), e.child_prompt) for e in terminals]
        values.append(float(np.mean(sims)))
    return values


def root_level_semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> Dict[str, float]:
    vals = root_level_semantic_preservation(edges, embedder, reference=reference)
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std":  float(np.std(vals))  if vals else 0.0,
    }


def semantic_preservation_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, Dict[str, float]]:
    return {name: semantic_preservation_stats(es, embedder) for name, es in grouped.items()}


def root_level_semantic_preservation_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, Dict[str, float]]:
    return {
        name: root_level_semantic_preservation_stats(es, embedder)
        for name, es in grouped.items()
    }


# ---------------------------------------------------------------------------
# Judge-Consistent Obfuscation (JCO)
# ---------------------------------------------------------------------------

def judge_consistent_obfuscation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> float:
    """Edge-level JCO: mean of success * cosine(reference, child)."""
    if not edges:
        return 0.0
    ref = lambda e: e.root_prompt if reference == "root" else e.parent_prompt
    scores = [float(e.success) * embedder.cosine(ref(e), e.child_prompt) for e in edges]
    return float(np.mean(scores))


def root_level_jco(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> float:
    """Root-level JCO: for each root take the best (highest SP) successful mutation."""
    if not edges:
        return 0.0
    by_root = group_edges_by_root(edges)
    ref = lambda e: e.root_prompt if reference == "root" else e.parent_prompt
    values = []
    for es in by_root.values():
        best = max(
            (embedder.cosine(ref(e), e.child_prompt) for e in es if e.success),
            default=0.0,
        )
        values.append(best)
    return float(np.mean(values))


def jco_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, float]:
    return {name: judge_consistent_obfuscation(es, embedder) for name, es in grouped.items()}


def root_level_jco_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, float]:
    return {name: root_level_jco(es, embedder) for name, es in grouped.items()}


# ---------------------------------------------------------------------------
# Mutation Efficiency Index (MEI)
# ---------------------------------------------------------------------------

def mutation_efficiency_index(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> float:
    """MEI = mean of success * SP / refinement_iter (zero for failures)."""
    if not edges:
        return 0.0
    ref = lambda e: e.root_prompt if reference == "root" else e.parent_prompt
    scores = [
        float(e.success) * embedder.cosine(ref(e), e.child_prompt) / max(e.refinement_iter, 1)
        for e in edges
    ]
    return float(np.mean(scores))


def mei_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, float]:
    return {name: mutation_efficiency_index(es, embedder) for name, es in grouped.items()}


# ---------------------------------------------------------------------------
# Depth of Success
# ---------------------------------------------------------------------------

def depth_of_success(edges: List[MutationEdge]) -> List[int]:
    return [e.refinement_iter for e in edges if e.success]


def root_level_depth_of_success(edges: List[MutationEdge]) -> List[int]:
    """Per-root minimum refinement_iter at which a success was achieved."""
    by_root = group_edges_by_root(edges)
    result = []
    for es in by_root.values():
        depths = [e.refinement_iter for e in es if e.success]
        if depths:
            result.append(min(depths))
    return result


def _depth_stats(depths: List[int]) -> Dict[str, float]:
    if not depths:
        return {}
    return {"mean": float(np.mean(depths)), "median": float(np.median(depths))}


def depth_of_success_per_operator(
    grouped: Dict[str, List[MutationEdge]],
) -> Dict[str, Dict[str, float]]:
    return {name: _depth_stats(depth_of_success(es)) for name, es in grouped.items()}


def root_level_depth_of_success_per_operator(
    grouped: Dict[str, List[MutationEdge]],
) -> Dict[str, Dict[str, float]]:
    return {
        name: _depth_stats(root_level_depth_of_success(es))
        for name, es in grouped.items()
    }


def global_depth_of_success(edges: List[MutationEdge]) -> Dict[str, float]:
    depths = depth_of_success(edges)
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max":  int(max(depths))       if depths else 0,
    }


def global_root_level_depth_of_success(edges: List[MutationEdge]) -> Dict[str, float]:
    depths = root_level_depth_of_success(edges)
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max":  int(max(depths))       if depths else 0,
    }


# ---------------------------------------------------------------------------
# Semantic Drift Rate
# ---------------------------------------------------------------------------

def compute_semantic_drift_rate(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
) -> Dict[str, float]:
    """
    Per-edge drift = sim(root, parent) - sim(root, child).
    For refinement_iter <= 1 the parent is the root so sim(root, parent) = 1.0.
    """
    drifts = []
    for e in edges:
        sim_parent = (
            1.0 if e.refinement_iter <= 1
            else embedder.cosine(e.root_prompt, e.parent_prompt)
        )
        sim_child = embedder.cosine(e.root_prompt, e.child_prompt)
        drifts.append(sim_parent - sim_child)
    return {
        "mean": float(np.mean(drifts)) if drifts else 0.0,
        "std":  float(np.std(drifts))  if drifts else 0.0,
    }


def drift_rate_per_operator(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
) -> Dict[str, Dict[str, float]]:
    return {name: compute_semantic_drift_rate(es, embedder) for name, es in grouped.items()}


# ---------------------------------------------------------------------------
# Cumulative metrics by depth
# ---------------------------------------------------------------------------

def cumulative_asr_by_depth(edges: List[MutationEdge]) -> Dict[int, float]:
    by_root = group_edges_by_root(edges)
    max_depth = max(e.refinement_iter for e in edges)
    return {
        d: sum(
            1 for es in by_root.values()
            if any(e.success and e.refinement_iter <= d for e in es)
        ) / len(by_root)
        for d in range(max_depth + 1)
    }


def cumulative_jco_by_depth(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
) -> Dict[int, float]:
    by_root = group_edges_by_root(edges)
    max_depth = max(e.refinement_iter for e in edges)
    result = {}
    for d in range(max_depth + 1):
        values = [
            embedder.cosine(e.root_prompt, e.child_prompt)
            for es in by_root.values()
            for e in es
            if e.refinement_iter <= d and e.success
        ]
        result[d] = float(np.mean(values)) if values else 0.0
    result[0] = 0.0
    return result


def cumulative_preservation_by_depth(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
) -> Dict[int, float]:
    by_root = group_edges_by_root(edges)
    max_depth = max(e.refinement_iter for e in edges)
    result = {}
    for d in range(max_depth + 1):
        values = [
            embedder.cosine(e.root_prompt, e.child_prompt)
            for es in by_root.values()
            for e in es
            if e.refinement_iter <= d
        ]
        result[d] = float(np.mean(values)) if values else 0.0
    result[0] = 1.0
    return result


def cumulative_metrics_by_depth(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
) -> Dict[int, Dict[str, float]]:
    asr  = cumulative_asr_by_depth(edges)
    jco  = cumulative_jco_by_depth(edges, embedder)
    pres = cumulative_preservation_by_depth(edges, embedder)
    return {
        d: {"asr": asr[d], "jco": jco[d], "preservation": pres[d]}
        for d in sorted(asr)
    }


# ---------------------------------------------------------------------------
# Collect all metrics into a results dict
# ---------------------------------------------------------------------------

def collect_results(
    edges: List[MutationEdge],
    grouped: Optional[Dict[str, List[MutationEdge]]],
    embedder: EmbeddingCache,
) -> dict:
    unique_roots = {e.root_prompt for e in edges}

    results: dict = {
        "global": {
            "num_edges":                           len(edges),
            "num_unique_root_prompts":             len(unique_roots),
            "edge_level_asr":                      attack_success_rate(edges),
            "root_level_asr":                      root_level_attack_success_rate(edges),
            "edge_level_semantic_preservation":    semantic_preservation_stats(edges, embedder),
            "root_level_semantic_preservation":    root_level_semantic_preservation_stats(edges, embedder),
            "edge_level_jco":                      judge_consistent_obfuscation(edges, embedder),
            "root_level_jco":                      root_level_jco(edges, embedder),
            "edge_level_depth_of_success":         global_depth_of_success(edges),
            "root_level_depth_of_success":         global_root_level_depth_of_success(edges),
            "mutation_efficiency_index":           mutation_efficiency_index(edges, embedder),
            "semantic_drift_rate":                 compute_semantic_drift_rate(edges, embedder),
        }
    }

    if grouped is not None:
        asr_per         = attack_success_rate_per_operator(grouped)
        root_asr_per    = root_level_asr_per_operator(grouped)
        sp_per          = semantic_preservation_per_operator(grouped, embedder)
        root_sp_per     = root_level_semantic_preservation_per_operator(grouped, embedder)
        jco_per         = jco_per_operator(grouped, embedder)
        root_jco_per    = root_level_jco_per_operator(grouped, embedder)
        depth_per       = depth_of_success_per_operator(grouped)
        root_depth_per  = root_level_depth_of_success_per_operator(grouped)
        mei_per         = mei_per_operator(grouped, embedder)
        drift_per       = drift_rate_per_operator(grouped, embedder)

        results["per_operator"] = {
            name: {
                "num_edges":                        len(es),
                "edge_level_asr":                   asr_per[name],
                "root_level_asr":                   root_asr_per[name],
                "edge_level_semantic_preservation": sp_per[name],
                "root_level_semantic_preservation": root_sp_per[name],
                "edge_level_jco":                   jco_per[name],
                "root_level_jco":                   root_jco_per[name],
                "edge_level_depth_of_success":      depth_per.get(name, {}),
                "root_level_depth_of_success":      root_depth_per.get(name, {}),
                "mutation_efficiency_index":        mei_per[name],
                "semantic_drift_rate":              drift_per[name],
            }
            for name, es in grouped.items()
        }

    return results


def save_results_json(results: dict, output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[RESULTS] Saved metrics to {p}")