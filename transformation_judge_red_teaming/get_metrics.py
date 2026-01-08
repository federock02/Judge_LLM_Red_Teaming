from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Iterable, Union
import hashlib
import json
from collections import defaultdict
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import seaborn as sns
import matplotlib.colors as mcolors

import pandas as pd

def build_transform_color_map(transform_names):
    palette = sns.color_palette("Paired", n_colors=len(transform_names)+1)
    return {name: palette[i] for i, name in enumerate(transform_names)} | {"Average": palette[-1]}

@dataclass(frozen=True)
class MutationEdge:
    # --------
    # identity
    # --------
    iteration_id: int
    refinement_iter: int
    depth: int

    # -------------
    # root context
    # -------------
    root_prompt: str
    root_fitness: float
    source_dataset: Optional[str]

    # ----------------
    # mutation context
    # ----------------
    parent_prompt: str
    parent_fitness: float

    child_prompt: str
    child_fitness: float
    delta_fitness: float
    success: bool

    # ------------------
    # operator metadata
    # ------------------
    operator_type: str              # e.g. "transformation", "persona"
    operator_name: str              # transformation name
    operator_metadata: Dict[str, Any]

    # ------------------
    # convenience flags
    # ------------------
    is_root_edge: bool              # True if parent_prompt == root_prompt


def load_mutation_edges_from_archive(
    archive_path: str,
    *,
    operator_type: str = "transformation",
    strict: bool = False
) -> List[MutationEdge]:
    """
    Load a red-teaming archive JSON and extract a flat list of MutationEdge objects.

    Args:
        archive_path: path to archive json
        operator_type: "transformation" or "persona"
        strict: if True, raises on malformed entries; otherwise skips them

    Returns:
        List[MutationEdge]
    """
    try:
        with open(archive_path, "r") as f:
            archive = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load archive from {archive_path}: {e}")
        return []

    edges: List[MutationEdge] = []

    for iter_key, iter_data in archive.items():
        try:
            iteration_id = int(iter_key)
        except ValueError:
            if strict:
                raise
            continue

        # ----------------
        # root-level info
        # ----------------
        root_prompt = iter_data.get("parent_prompt")
        root_fitness = iter_data.get("parent_fitness")

        parent_metadata = iter_data.get("parent_metadata", {})
        source_dataset = parent_metadata.get("source")

        operator_info = iter_data.get("transformation", {})
        operator_name = operator_info.get("name", "unknown")
        operator_metadata = dict(operator_info)

        attempts = iter_data.get("attempts", [])
        if not isinstance(attempts, list):
            if strict:
                raise ValueError("attempts is not a list")
            continue

        # -------------------
        # attempt-level walk
        # -------------------
        for attempt in attempts:
            try:
                refinement_iter = int(attempt.get("refinement_iter"))
                depth = refinement_iter + 1

                parent_prompt = attempt.get("parent_prompt")
                parent_fitness = attempt.get("parent_score")

                mutated_prompts = attempt.get("mutated_prompts", [])
                fitness = attempt.get("fitness", [])
                delta_fitness = attempt.get("delta_fitness", [])
                success_flags = attempt.get("success", [])

                # defensive length check
                n = min(
                    len(mutated_prompts),
                    len(fitness),
                    len(delta_fitness),
                    len(success_flags),
                )

                if n == 0:
                    continue

                for i in range(n):
                    child_prompt = mutated_prompts[i]
                    child_fitness = float(fitness[i])
                    delta = float(delta_fitness[i])
                    success = bool(success_flags[i])

                    edge = MutationEdge(
                        iteration_id=iteration_id,
                        refinement_iter=refinement_iter,
                        depth=depth,

                        root_prompt=root_prompt,
                        root_fitness=float(root_fitness),
                        source_dataset=source_dataset,

                        parent_prompt=parent_prompt,
                        parent_fitness=float(parent_fitness),

                        child_prompt=child_prompt,
                        child_fitness=child_fitness,
                        delta_fitness=delta,
                        success=success,

                        operator_type=operator_type,
                        operator_name=operator_name,
                        operator_metadata=operator_metadata,

                        is_root_edge=(parent_prompt == root_prompt),
                    )

                    edges.append(edge)

            except Exception:
                if strict:
                    raise
                continue

    return edges

def load_mutation_edges_from_archives(
    archive_paths: Union[str, Iterable[str]],
    *,
    operator_type: str = "transformation",
    strict: bool = False,
) -> list[MutationEdge]:
    """
    Load and merge multiple red-teaming archive JSONs into a single edge list.

    Args:
        archive_paths:
            - list of paths
            - directory
            - glob pattern (e.g. results/*.json)
        operator_type:
            "transformation" or "persona"
        strict:
            fail fast on malformed archives
        deduplicate:
            remove identical mutation edges across runs

    Returns:
        Flat list of MutationEdge with run_id populated
    """
    # ---------------------
    # resolve archive paths
    # ---------------------
    if isinstance(archive_paths, str):
        p = Path(archive_paths)
        if p.is_dir():
            paths = sorted(p.glob("*.json"))
        else:
            paths = sorted(Path().glob(archive_paths))
    else:
        paths = [Path(p) for p in archive_paths]

    all_edges: list[MutationEdge] = []

    for path in paths:
        run_id = path.stem  # stable, human-readable

        edges = load_mutation_edges_from_archive(
            archive_path=str(path),
            operator_type=operator_type,
            strict=strict,
        )

        all_edges.extend(edges)

    return all_edges


def group_edges_by_operator(
    edges: List[MutationEdge],
) -> Dict[str, List[MutationEdge]]:
    grouped: Dict[str, List[MutationEdge]] = defaultdict(list)
    for e in edges:
        grouped[e.operator_name].append(e)
    return grouped

def group_edges_by_root(edges):
    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)
    return by_root


class EmbeddingCache:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self.cache[text]

    def cosine(self, a: str, b: str) -> float:
        va = self.embed(a)
        vb = self.embed(b)
        return float(np.dot(va, vb))


def plot_cumulative_metrics_vs_depth(
    depth_stats: dict,
    vis_dir: str,
):
    depths = sorted(depth_stats.keys())

    asr_vals = [depth_stats[d]["asr"] for d in depths]
    pres_vals = [depth_stats[d]["preservation"] for d in depths]
    jco_vals = [depth_stats[d]["jco"] for d in depths]
    print(asr_vals)
    print(pres_vals)
    print(jco_vals)

    plt.figure(figsize=(12, 8))

    plt.plot(depths, asr_vals, marker="o", linewidth=3, label="Cumulative ASR")
    plt.plot(depths, pres_vals, marker="s", linewidth=3, label="Cumulative Semantic Preservation")
    plt.plot(depths, jco_vals, marker="^", linewidth=3, label="Cumulative JCO")

    plt.xlabel("Refinement Depth", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.title("Cumulative Attack Metrics vs Refinement Depth", fontsize=24)

    plt.xticks(depths, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "cumulative_metrics_vs_depth.png"),
        dpi=200,
    )
    plt.close()


def plot_metrics_vs_depth(
    depth_stats: Dict[int, Dict[str, float]],
    vis_dir: str,
):
    depths = sorted(depth_stats.keys())

    asr_vals = [depth_stats[d]["asr"] for d in depths]
    pres_vals = [depth_stats[d]["preservation"] for d in depths]
    jco_vals = [depth_stats[d]["jco"] for d in depths]

    plt.figure(figsize=(12, 8))

    plt.plot(
        depths, asr_vals,
        marker="o",
        linewidth=3,
        label="Attack Success Rate (ASR)",
    )
    plt.plot(
        depths, pres_vals,
        marker="s",
        linewidth=3,
        label="Semantic Preservation (SP)",
    )
    plt.plot(
        depths, jco_vals,
        marker="^",
        linewidth=3,
        label="Judge-Consistent Obfuscation (JCO)",
    )

    plt.xlabel("Refinement Depth", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.title("Evolution of Metrics Across Refinement Depth", fontsize=24)

    plt.xticks(depths, fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "metrics_vs_depth.png"),
        dpi=200,
    )
    plt.close()


def cumulative_asr_by_depth(edges):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    asr = {}
    for d in range(max_depth + 1):
        successes = 0
        for root, es in by_root.items():
            if any(e.success and e.depth <= d for e in es):
                successes += 1
        asr[d] = successes / len(by_root)

    return asr


def cumulative_jco_by_depth(edges, embedder):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    jco = {}
    for d in range(max_depth + 1):
        values = []
        for root, es in by_root.items():
            best = 0.0
            for e in es:
                if e.depth <= d and e.success:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    best = max(best, sim)
                    values.append(sim)
                #values.append(best)
        jco[d] = float(np.mean(values))
    jco[0] = 0.0  # at depth 0, JCO is 0.0

    return jco


def cumulative_preservation_by_depth(edges, embedder):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    preservation = {}
    for d in range(max_depth + 1):
        values = []
        """
        for root, es in by_root.items():
            best = 0.0
            for e in es:
                if e.depth == d:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    best = max(best, sim)
            values.append(best)
        preservation[d] = float(np.mean(values))
        """
        for root, es in by_root.items():
            for e in es:
                if e.depth <= d:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    values.append(sim)
        preservation[d] = float(np.mean(values))
    preservation[0] = 1.0  # at depth 0, preservation is perfect

    return preservation


def cumulative_metrics_by_depth(edges, embedder):
    asr = cumulative_asr_by_depth(edges)
    jco = cumulative_jco_by_depth(edges, embedder)
    preservation = cumulative_preservation_by_depth(edges, embedder)

    depths = sorted(asr.keys())
    stats = {}

    for d in depths:
        print("depth: ", d)
        stats[d] = {
            "asr": asr[d],
            "jco": jco[d],
            "preservation": preservation[d],
        }
    
    print(stats)

    return stats


def global_semantic_preservation(edges, embedder, reference="root"):
    preservations = semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)) if preservations else 0.0,
        "std": float(np.std(preservations)) if preservations else 0.0,
    }

def global_root_level_semantic_preservation(edges, embedder, reference="root"):
    preservations = root_level_semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)) if preservations else 0.0,
        "std": float(np.std(preservations)) if preservations else 0.0,
    }


def global_depth_of_success(edges):
    depths = [e.depth for e in edges if e.success]
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max": int(max(depths)) if depths else 0,
    }

def global_root_level_depth_of_success(edges):
    by_root = group_edges_by_root(edges)
    depths = []
    for root, es in by_root.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append(min(successful_depths))
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max": int(max(depths)) if depths else 0,
    }


def collect_results(
    edges,
    grouped_edges,
    embedder,
):
    results = {}
    unique_root_set = {e.root_prompt for e in edges}

    # ----------------
    # Global statistics
    # ----------------
    results["global"] = {
        "num_edges": len(edges),
        "num_total_mutations": len(edges),
        "num_unique_root_prompts": len(unique_root_set),
        "edge_level_attack_success_rate": attack_success_rate(edges),
        "edge_level_semantic_preservation": global_semantic_preservation(edges, embedder),
        "edge_level_judge_consistent_obfuscation": judge_consistent_obfuscation(edges, embedder),
        "edge_level_depth_of_success": global_depth_of_success(edges),
        "root_level_attack_success_rate": root_level_attack_success_rate(edges),
        "root_level_semantic_preservation": global_root_level_semantic_preservation(edges, embedder),
        "root_level_judge_consistent_obfuscation": root_level_jco(edges, embedder),
        "root_level_depth_of_success": global_root_level_depth_of_success(edges),
        "mutation_efficiency_index": mutation_efficiency_index(edges, embedder),
        "semantic_drift_rate": compute_semantic_drift_rate(edges, embedder)
    }

    # -------------------------
    # Per-transformation stats
    # -------------------------
    per_transform = {}

    asr_per = attack_success_rate_per_transform(grouped_edges)
    root_asr_per = root_level_asr_per_transform(grouped_edges)
    preservation_per = semantic_preservation_per_transform(grouped_edges, embedder)
    root_preservation_per = root_level_semantic_preservation_per_transform(grouped_edges, embedder)
    jco_per = jco_per_transform(grouped_edges, embedder)
    root_jco_per = root_level_jco_per_transform(grouped_edges, embedder)
    depth_per = depth_of_success_per_transform(grouped_edges)
    root_depth_per = root_level_depth_of_success_per_transform(grouped_edges)
    mei = mei_per_transform(grouped_edges, embedder)
    semantic_drift = drift_rate_per_transform(grouped_edges, embedder)


    for name, edges_t in grouped_edges.items():
        per_transform[name] = {
            "num_edges": len(edges_t),
            "global_attack_success_rate": asr_per[name],
            "root_level_attack_success_rate": root_asr_per[name],
            "global_semantic_preservation": preservation_per[name],
            "root_level_semantic_preservation": root_preservation_per[name],
            "global_judge_consistent_obfuscation": jco_per[name],
            "root_level_judge_consistent_obfuscation": root_jco_per[name],
            "global_depth_of_success": depth_per[name],
            "root_level_depth_of_success": root_depth_per[name],
            "mutation_efficiency_index": mei[name],
            "semantic_drift_rate": semantic_drift[name],
        }

    results["per_transformation"] = per_transform

    return results

def save_results_json(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[RESULTS] Saved metrics to {output_path}")

# -------------------------
# ------- METRICS ---------
# -------------------------
# -- Attack Success Rate -- 
def attack_success_rate(
    edges: List[MutationEdge],
) -> float:
    if not edges:
        return 0.0
    return np.mean([e.success for e in edges])

# -- Attack Success Rate per Transformation --
def attack_success_rate_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {
        name: attack_success_rate(edges)
        for name, edges in grouped_edges.items()
    }

def plot_asr(asr_per_transform: Dict[str, float], vis_dir: str, color_map):
    names = list(asr_per_transform.keys())
    values = list(asr_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("ASR_G", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.grid(axis="y", alpha=0.3)
    plt.yticks(fontsize=18)
    plt.title("Global Attack Success Rate per Transformation (ASR_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_attack_success_rate_per_transform.png"))
    plt.close()

# -- Root-Level Attack Success Rate --
def root_level_attack_success_rate(
    edges: List[MutationEdge],
) -> float:
    if not edges:
        return 0.0

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)

    successes = 0
    for root, es in by_root.items():
        if any(e.success for e in es): # counting if at least one successful mutation per root
            successes += 1

    return successes / len(by_root)

# -- Root-Level ASR per Transformation --
def root_level_asr_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {
        name: root_level_attack_success_rate(edges)
        for name, edges in grouped_edges.items()
    }

# -- Plot Root-Level ASR --
def plot_root_level_asr(
    root_asr_per_transform: Dict[str, float],
    vis_dir: str,
    color_map,
):
    names = list(root_asr_per_transform.keys())
    values = list(root_asr_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("ASR_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)

    plt.title(
        "Root-level Attack Success Rate per Transformation (ASR_R)",
        fontsize=24,
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "root_level_attack_success_rate_per_transform.png")
    )
    plt.close()

# -- Semantic Preservation --
def semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",  # "parent" or "root"
) -> List[float]:
    preservations = []

    for e in edges:
        ref_text = e.parent_prompt if reference == "parent" else e.root_prompt
        sim = embedder.cosine(ref_text, e.child_prompt)
        preservations.append(sim)

    return preservations

# -- Semantic Preservation Stats --
def semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
):
    preservations = semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)),
        "std": float(np.std(preservations)),
    }

# -- Semantic Preservation per Transformation --
def semantic_preservation_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: semantic_preservation_stats(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_semantic_preservation(preservation_per_transform: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(preservation_per_transform.keys())
    means = [v["mean"] for v in preservation_per_transform.values()]
    stds = [v["std"] for v in preservation_per_transform.values()]
    colors = [color_map[n] for n in names]

    mean = np.mean(means)
    names.append("AVERAGE")
    means.append(mean)
    stds.append(np.std(means))
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, yerr=stds, capsize=4, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("SP_G", fontsize=22)
    plt.yticks(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.title("Global Semantic Preservation per Transformation (SP_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_semantic_preservation_per_transform.png"))
    plt.close()

def plot_semantic_preservation_violin(
    grouped_edges,
    embedder,
    vis_dir,
    color_map,
    reference="root",
):
    data = []
    labels = []

    for name, edges in grouped_edges.items():
        preservations = semantic_preservation(edges, embedder, reference=reference)
        data.append(preservations)
        labels.append(name)

    plt.figure(figsize=(24, 10))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_map[labels[i]])
        pc.set_alpha(0.7)

    plt.grid(axis="y", alpha=0.3)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=22)
    plt.ylabel("SP_G", fontsize=24)
    plt.yticks(fontsize=20)
    plt.title("Global Semantic Preservation Distribution per Transformation (SP_G)", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_semantic_preservation_violin.png"))
    plt.close()


# -- Root-Level Semantic Preservation --
def root_level_semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> List[float]: # Return a list, not a float
    if not edges:
        return []

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)
    
    print("Number of unique roots: ", len(by_root))

    values = []
    for root, es in by_root.items():
        val = []
        # To match your report: find the terminal mutation (highest depth)
        # regardless of success, or use your success logic if preferred.
        terminal_edges = [e for e in es if e.success or e.depth == max(ev.depth for ev in es)]
        
        for terminal_edge in terminal_edges:
            ref = terminal_edge.parent_prompt if reference == "parent" else terminal_edge.root_prompt
            sim = embedder.cosine(ref, terminal_edge.child_prompt)
            values.append(sim)

        #values.append(np.mean(val))  # Average if multiple terminal edges per root
        
    return values

# -- Root-Level Semantic Preservation Stats --
def root_level_semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    preservations = root_level_semantic_preservation(edges, embedder, reference=reference)
    
    if not preservations:
        return {"mean": 0.0, "std": 0.0}
        
    return {
        "mean": float(np.mean(preservations)),
        "std": float(np.std(preservations)),
    }

# -- Root-Level Semantic Preservation per Transformation --
def root_level_semantic_preservation_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
):
    return {
        name: root_level_semantic_preservation_stats(edges, embedder, reference=reference)
        for name, edges in grouped_edges.items()
    }


def plot_root_level_semantic_preservation(
    root_preservation_per_transform: Dict[str, Dict[str, float]], # Correct type hint
    vis_dir: str,
    color_map,
):
    names = list(root_preservation_per_transform.keys())
    means = [v["mean"] for v in root_preservation_per_transform.values()]
    stds = [v["std"] for v in root_preservation_per_transform.values()]
    
    # Ensure color_map key exists (using .get() or checking previous logic)
    colors = [color_map.get(n, "blue") for n in names]

    global_mean = np.mean(means) if means else 0.0
    names.append("AVERAGE")
    means.append(global_mean)
    stds.append(np.std(means) if len(means) > 1 else 0.0)
    
    # Use the same color logic for average as your other plots
    colors.append(color_map.get("Average", color_map.get(-1, "black")))

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, yerr=stds, capsize=4, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("SP_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)
    plt.title("Root-level Semantic Preservation per Transformation (SP_R)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_semantic_preservation_per_transform.png"))
    plt.close()


# -- Judge Consistent Obfuscation (JCO) --
def judge_consistent_obfuscation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    scores = []
    for e in edges:
        sim = embedder.cosine(e.parent_prompt, e.child_prompt) if reference == "parent" else embedder.cosine(e.root_prompt, e.child_prompt) # JCO relative to root or parent
        jco = float(e.success) * sim
        scores.append(jco)
    return np.mean(scores) if scores else 0.0

# -- JCO per Transformation --
def jco_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: judge_consistent_obfuscation(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_jco(jco_scores: Dict[str, float], vis_dir: str, color_map):
    names = list(jco_scores.keys())
    values = list(jco_scores.values())
    colors = [color_map[n] for n in names]

    global_jco = np.mean(values)
    values.append(global_jco)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("JCO_G", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(np.arange(0, 0.5, 0.1), fontsize=18)
    plt.title("Global JCO per Transformation (JCO_G)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_jco_per_transform.png"))
    plt.close()


# -- Root-Level JCO --
def root_level_jco(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> float:
    if not edges:
        return 0.0

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)

    values = []
    for root, es in by_root.items():
        best = 0.0
        for e in es:
            if not e.success:
                continue
            ref = e.parent_prompt if reference == "parent" else e.root_prompt
            sim = embedder.cosine(ref, e.child_prompt)
            best = max(best, sim)
        values.append(best)

    return float(np.mean(values))


# -- Root-Level JCO per Transformation --
def root_level_jco_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
):
    return {
        name: root_level_jco(edges, embedder, reference=reference)
        for name, edges in grouped_edges.items()
    }


def plot_root_level_jco(
    root_jco_per_transform: Dict[str, float],
    vis_dir: str,
    color_map,
):
    names = list(root_jco_per_transform.keys())
    values = list(root_jco_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("JCO_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(np.arange(0, 0.8, 0.1),fontsize=18)
    plt.title("Root-level JCO per Transformation (JCO_R)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_jco_per_transform.png"))
    plt.close()

# -- Mutation Efficiency Index (MEI) --
def mutation_efficiency_index(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    scores = []
    for e in edges:
        sim = embedder.cosine(e.parent_prompt, e.child_prompt) if reference == "parent" else embedder.cosine(e.root_prompt, e.child_prompt) # JCO relative to root or parent
        mei = float(e.success) * sim / e.depth
        scores.append(mei)
    return np.mean(scores) if scores else 0.0

# -- MEI per Transformation --
def mei_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: mutation_efficiency_index(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_mei(mei_scores: Dict[str, float], vis_dir: str, color_map):
    names = list(mei_scores.keys())
    values = list(mei_scores.values())
    colors = [color_map[n] for n in names]

    global_mei = np.mean(values)
    values.append(global_mei)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("MEI", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)
    plt.title("Mutation Efficiency Index per Transformation (MEI)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "mei_per_transform.png"))
    plt.close()


def plot_asr_preservation_heatmap(
    asr_per_transform,
    preservation_per_transform,
    vis_dir,
    *,
    type="edge", # "edge" or "root"
):
    sns.set_context("talk", font_scale=2.2)

    rows = []
    for name in asr_per_transform:
        rows.append([
            name,
            asr_per_transform[name],
            preservation_per_transform[name]["mean"],
        ])
    rows.append([
        "AVERAGE",
        np.mean(list(asr_per_transform.values())),
        np.mean([v["mean"] for v in preservation_per_transform.values()]),
    ])

    df = pd.DataFrame(
        rows,
        columns=["Transformation", "ASR", "SP"],
    ).set_index("Transformation")

    plt.figure(
        figsize=(16, max(6, 0.9 * len(df))),
        dpi=200,
    )

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 26},
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "Global ASR and SP per Transformation" if type == "edge" else "Root-level ASR and SP per Transformation",
        fontsize=32,
        pad=25,
    )

    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(
        os.path.join(vis_dir, f"{type}_asr_preservation_heatmap.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_asr_vs_preservation(
    asr_per_transform,
    preservation_per_transform,
    vis_dir,
    color_map,
    *,
    type="edge", # "edge" or "root"
):
    plt.figure(figsize=(12, 10))

    asr_per_transform["Average"] = np.mean(list(asr_per_transform.values()))
    preservation_per_transform["Average"] = {
        "mean": np.mean([v["mean"] for v in preservation_per_transform.values()])
    }

    for name in asr_per_transform.keys():
        if name == "Average":
            continue  # skip average, will plot separately
        if name not in preservation_per_transform:
            continue  # skip if missing in preservation dict
        plt.scatter(
            preservation_per_transform[name]["mean"],
            asr_per_transform[name],
            color=color_map.get(name, "gray"),
            label=name,
            s=300,
        )
    plt.scatter(
        preservation_per_transform["Average"]["mean"],
        asr_per_transform["Average"],
        color=color_map["Average"],
        label="AVERAGE",
        s=450,
    )
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Semantic Preservation (⟶ better)", fontsize=20)
    plt.ylabel("Attack Success Rate (⟶ better)", fontsize=20)
    plt.title("ASR vs Semantic Preservation", fontsize=22)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{type}_asr_vs_preservation.png"))
    plt.close()

# -- Depth of Success --
def depth_of_success(edges: List[MutationEdge]) -> List[int]:
    depths = []
    for e in edges:
        if e.success:
            depths.append(e.depth)
    return depths

# -- Depth of Success per Transformation --
def depth_of_success_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
):
    stats = {}
    for name, edges in grouped_edges.items():
        depths = depth_of_success(edges)
        if depths:
            stats[name] = {
                "mean": np.mean(depths),
                "median": np.median(depths),
            }
    return stats

def plot_depth_of_success(depth_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(depth_stats.keys())
    means = [v["mean"] for v in depth_stats.values()]
    colors = [color_map[n] for n in names]

    global_depth = np.mean(means)
    means.append(global_depth)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("d_G", fontsize=22)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks([0,1,2,3,4], fontsize=18)
    plt.title("Global Depth of Success per Transformation (d_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_depth_of_success_per_transform.png"))
    plt.close()

# -- Root-Level Depth of Success --
def root_level_depth_of_success(edges: List[MutationEdge]) -> List[int]:
    depths = []
    root_edges = group_edges_by_root(edges)
    for root, es in root_edges.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append(min(successful_depths))
    return depths

# -- Root-Level Depth of Success per Transformation --
def root_level_depth_of_success_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
):
    stats = {}
    for name, edges in grouped_edges.items():
        depths = root_level_depth_of_success(edges)
        if depths:
            stats[name] = {
                "mean": np.mean(depths),
                "median": np.median(depths),
            }
    return stats

def plot_root_level_depth_of_success(depth_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(depth_stats.keys())
    means = [v["mean"] for v in depth_stats.values()]
    colors = [color_map[n] for n in names]

    global_depth = np.mean(means)
    means.append(global_depth)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)

    plt.ylabel("d_R", fontsize=22)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(np.arange(0, 2.0, 0.5),fontsize=18)
    plt.title("Root-Level Depth of Success per Transformation (d_R)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_depth_of_success_per_transform.png"))
    plt.close()


# -- Drift rate --
def compute_semantic_drift_rate(edges: List[MutationEdge], embedder: EmbeddingCache) -> Dict[str, float]:
    """
    Calculates the average semantic drift (loss of SP) per mutation step.
    """
    drifts = []
    
    for e in edges:
        # Similarity of Parent to Root (If depth 1, parent is root, so sim is 1.0)
        if e.depth <= 1:
            sim_parent_root = 1.0
        else:
            sim_parent_root = embedder.cosine(e.root_prompt, e.parent_prompt)
            
        # Similarity of Child to Root
        sim_child_root = embedder.cosine(e.root_prompt, e.child_prompt)
        
        # Drift = Loss in preservation
        drift = sim_parent_root - sim_child_root
        drifts.append(drift)
        
    return {
        "mean": float(np.mean(drifts)) if drifts else 0.0,
        "std": float(np.std(drifts)) if drifts else 0.0
    }

def drift_rate_per_transform(grouped_edges, embedder):
    return {
        name: compute_semantic_drift_rate(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_drift_rate(drift_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(drift_stats.keys())
    means = [v["mean"] for v in drift_stats.values()]
    stds = [v["std"] for v in drift_stats.values()]
    colors = [color_map[n] for n in names]

    global_drift = np.mean(means)
    means.append(global_drift)
    stds.append(np.std(means))
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    # Higher bar = more drift (meaning lost faster)
    plt.bar(names, means, yerr=stds, capsize=5, color=colors)

    plt.axvline(x=len(names) - 1.5, color='black', linestyle='--', linewidth=2, alpha=0.6)
    
    plt.grid(axis="y", alpha=0.3)
    plt.ylabel("ΔSP", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)
    plt.title("Average Semantic Drift per Transformation (ΔSP)", fontsize=24)
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "semantic_drift_rate_per_transform.png"))
    plt.close()

# -- Best Delta Fitness per Iteration --
def best_delta_per_iteration(edges: List[MutationEdge]) -> Dict[int, float]:
    best = defaultdict(lambda: -np.inf)
    for e in edges:
        best[e.iteration_id] = max(best[e.iteration_id], e.delta_fitness)
    return dict(best)

def plot_best_delta(best_deltas: Dict[int, float], vis_dir: str):
    xs = sorted(best_deltas.keys())
    ys = [best_deltas[x] for x in xs]

    plt.figure(figsize=(16, 8))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Best Fitness Improvement", fontsize=16)
    plt.title("Best Attack Gain per Iteration", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "best_delta_per_iteration.png"))
    plt.close()

def main_analysis(archive_path: List[str] | str, vis_dir: str):
    edges = load_mutation_edges_from_archives(archive_path)
    grouped = group_edges_by_operator(edges)

    embedder = EmbeddingCache()

    transform_names = list(grouped.keys())
    color_map = build_transform_color_map(transform_names)

    # ---- Root-level semantic preservation
    root_preservation_per_transform = root_level_semantic_preservation_per_transform(grouped, embedder)
    plot_root_level_semantic_preservation(root_preservation_per_transform, vis_dir, color_map)
    print(f"[ANALYSIS] Completed SP_R plotting.")


    # collect metrics
    results = collect_results(
        edges=edges,
        grouped_edges=grouped,
        embedder=embedder,
    )

    # save
    save_results_json(
        results,
        output_path=f"{vis_dir}/transformation_attack_metrics.json",
    )
    return
    
    # ---- Edge-level ASR
    asr_per_transform = attack_success_rate_per_transform(grouped)
    plot_asr(asr_per_transform, vis_dir, color_map)
    print(f"[ANALYSIS] Completed ASR_G plotting.")

    # ---- Root-level ASR
    root_asr_per_transform = root_level_asr_per_transform(grouped)
    plot_root_level_asr(root_asr_per_transform, vis_dir, color_map)
    print(f"[ANALYSIS] Completed ASR_R plotting.")

    # ---- Semantic preservation
    preservation_per_transform = semantic_preservation_per_transform(grouped, embedder)
    plot_semantic_preservation(preservation_per_transform, vis_dir, color_map)
    # plot_semantic_preservation_violin(grouped, embedder, vis_dir, color_map)
    print(f"[ANALYSIS] Completed SP_G plotting.")

    # ---- Root-level semantic preservation
    root_preservation_per_transform = root_level_semantic_preservation_per_transform(grouped, embedder)
    plot_root_level_semantic_preservation(root_preservation_per_transform, vis_dir, color_map)
    print(f"[ANALYSIS] Completed SP_R plotting.")

    # ---- Heatmap
    plot_asr_preservation_heatmap(asr_per_transform, preservation_per_transform, vis_dir, type="edge")
    plot_asr_preservation_heatmap(root_asr_per_transform, root_preservation_per_transform, vis_dir, type="root")
    print(f"[ANALYSIS] Completed ASR vs SP heatmap plotting.")
    return

    # ---- Scatter
    plot_asr_vs_preservation(asr_per_transform, preservation_per_transform, vis_dir, color_map, type="edge")
    plot_asr_vs_preservation(root_asr_per_transform, root_preservation_per_transform, vis_dir, color_map, type="root")
    print(f"[ANALYSIS] Completed ASR vs SP scatter plotting.")

    # ---- JCO
    jco_scores = jco_per_transform(grouped, embedder)
    plot_jco(jco_scores, vis_dir, color_map)
    print(f"[ANALYSIS] Completed JCO_G plotting.")

    # ---- Root-level JCO
    root_jco_scores = root_level_jco_per_transform(grouped, embedder)
    plot_root_level_jco(root_jco_scores, vis_dir, color_map)
    print(f"[ANALYSIS] Completed JCO_R plotting.")

    # ---- MEI
    mei_scores = mei_per_transform(grouped, embedder)
    plot_mei(mei_scores, vis_dir, color_map)
    print(f"[ANALYSIS] Completed MEI plotting.")

    # ---- Depth
    depth_stats = depth_of_success_per_transform(grouped)
    plot_depth_of_success(depth_stats, vis_dir, color_map)
    print(f"[ANALYSIS] Completed Depth of Success plotting.")

    # ---- Root-level Depth
    root_depth_stats = root_level_depth_of_success_per_transform(grouped)
    plot_root_level_depth_of_success(root_depth_stats, vis_dir, color_map)
    print(f"[ANALYSIS] Completed Root-Level Depth of Success plotting.")

    # ---- Metrics vs Depth
    depth_stats = cumulative_metrics_by_depth(edges, embedder)
    plot_cumulative_metrics_vs_depth(depth_stats, vis_dir)
    print(f"[ANALYSIS] Completed Metrics vs Depth plotting.")

    # ---- Drift rate
    drift_stats = drift_rate_per_transform(grouped, embedder)
    plot_drift_rate(drift_stats, vis_dir, color_map)
    print(f"[ANALYSIS] Completed Semantic Drift Rate plotting.")

    # collect metrics
    results = collect_results(
        edges=edges,
        grouped_edges=grouped,
        embedder=embedder,
    )

    # save
    save_results_json(
        results,
        output_path=f"{vis_dir}/transformation_attack_metrics.json",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Red-Teaming Archive Metrics")
    parser.add_argument(
        "--results_archive",
        type=str,
        default=None,
        help="Path to the red-teaming json archive"
    )
    parser.add_argument(
        "--results_folders", 
        type=str, 
        nargs='+',  # '+' means 1 or more arguments; '*' means 0 or more
        default=None, 
        help="List of folders separated by spaces"
    )
    args = parser.parse_args()

    archive_path = os.path.abspath(args.results_archive) if args.results_archive else None
    result_folders = args.results_folders

    if archive_path is not None:
        print(f"[ANALYSIS] Loading archive from: {archive_path}")

        # visualization directory
        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        main_analysis(archive_path, vis_dir)
    else:
        paths = []
        for folder in result_folders:
            try:
                p = os.path.abspath(os.path.join(folder, "red_teaming_archive.json"))
                paths.append(p)
            except Exception as e:
                print(f"Failed to resolve archive in folder {folder}: {e}")
        print(f"[ANALYSIS] Loading archives from: {paths}")
        vis_dir = "visualizations_combined"
        os.makedirs(vis_dir, exist_ok=True)
        
        main_analysis(paths, vis_dir)