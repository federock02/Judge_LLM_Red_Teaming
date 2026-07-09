from __future__ import annotations
import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import numpy as np

# Assuming these are imported from your metrics.py
try:
    from persona_judge_red_teaming.metrics import (
        EmbeddingCache,
        MutationEdge,
        cumulative_metrics_by_depth
    )
except ImportError:
    # Fallback placeholders for script running independent verification tests
    class EmbeddingCache:
        def __init__(self, device): pass
        def batch_embed(self, texts): pass
    class MutationEdge:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)

def load_mutation_edges_from_archive(archive_path: str) -> List[MutationEdge]:
    try:
        with open(archive_path, "r") as f:
            archive = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {archive_path}: {e}")
        return []

    edges: List[MutationEdge] = []
    for iter_key, iter_data in archive.items():
        if iter_key.startswith("_"): continue
        try:
            iteration_id = int(iter_key)
            root_prompt = iter_data.get("parent_prompt")
            root_fitness = float(iter_data.get("parent_fitness", 0.0))
            source = iter_data.get("parent_metadata", {}).get("source")
            op_info = iter_data.get("transformation", {})
            op_name = op_info.get("name", "unknown")

            for attempt in iter_data.get("attempts", []):
                ref_iter = int(attempt["refinement_iter"])
                parent_p = attempt["parent_prompt"]
                parent_f = float(attempt["parent_score"])
                
                mutated = attempt.get("mutated_prompts", [])
                fitness = attempt.get("fitness", [])
                successes = attempt.get("success", [])

                print(f"[DEBUG] Iter {iteration_id}, Ref {ref_iter}, Parent Fit {parent_f}, Mutations {len(mutated)}")

                for i in range(len(mutated)):
                    edges.append(MutationEdge(
                        iteration_id=iteration_id,
                        refinement_iter=ref_iter,
                        root_prompt=root_prompt,
                        root_fitness=root_fitness,
                        source_dataset=source,
                        parent_prompt=parent_p,
                        parent_fitness=parent_f,
                        child_prompt=mutated[i],
                        child_fitness=float(fitness[i]),
                        delta_fitness=float(fitness[i]) - parent_f,
                        success=bool(successes[i]),
                        operator_type="mutation",
                        operator_name=op_name,
                        operator_metadata=op_info,
                        is_root_edge=(parent_p == root_prompt)
                    ))
        except Exception as e:
            print(f"[DEBUG] Skipping iteration key {iter_key} due to structural variation: {e}")
            continue
    return edges

def main():
    parser = argparse.ArgumentParser(description="Generate double-column plots with caching.")
    parser.add_argument("--cache_file", type=str, default="combined_visualizations/metrics_cache.json",
                        help="Path to save or load computed metrics JSON.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Skip computation and load metrics directly from cache file if it exists.")
    args = parser.parse_args()

    approaches = ["persona", "transformation"]
    judges = ["LlamaGuard", "ShieldGemma", "WildGuard"]
    
    all_stats = {app: {j: {} for j in judges} for app in approaches}
    cache_path = Path(args.cache_file)
    loaded_from_cache = False

    # Check if we can reuse computed metrics
    if args.use_cache:
        if cache_path.exists():
            print(f"[INFO] Loading metrics directly from cache: {cache_path}")
            try:
                with open(cache_path, "r") as f:
                    all_stats = json.load(f)
                loaded_from_cache = True
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}. Proceeding with clean computation.")
        else:
            print(f"[WARNING] Cache file {cache_path} not found. Computing from scratch.")

    if not loaded_from_cache:
        global_all_texts = set()
        all_edges_dict = {app: {j: [] for j in judges} for app in approaches}

        print("--- STEP 1: Discovering Archives and Loading Edges ---")
        for app in approaches:
            for judge in judges:
                base_search = f"{app}_judge_red_teaming/results/{judge}/*/red_teaming_archive.json"
                archive_paths = glob.glob(base_search)

                print(f"[{app.upper()} - {judge}] Found {len(archive_paths)} archives.")
                
                for path in archive_paths:
                    edges = load_mutation_edges_from_archive(path)
                    all_edges_dict[app][judge].extend(edges)
                    
                    for e in edges:
                        global_all_texts.update([e.root_prompt, e.parent_prompt, e.child_prompt])

        print(f"\n--- STEP 2: Warmup Embedding Cache ({len(global_all_texts)} texts) ---")
        embedder = EmbeddingCache(device="cuda:0")
        embedder.batch_embed(list(global_all_texts))

        print("\n--- STEP 3: Computing Metrics ---")
        for app in approaches:
            for judge in judges:
                edges = all_edges_dict[app][judge]
                if not edges: continue
                
                print(f"Processing {app} -> {judge}...")
                stats = cumulative_metrics_by_depth(edges, embedder)
                all_stats[app][judge] = {str(k): v for k, v in stats.items()}

        # Save metrics to cache file
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w") as f:
                json.dump(all_stats, f, indent=4)
            print(f"[INFO] Successfully saved computed metrics to cache: {cache_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write cache file: {e}")

    print("\n--- STEP 4: Generating Combined Dual-Axis Plot (Optimized for Double Column) ---")
    
    # Ultra scale-up configuration specifically designed for high-reduction templates
    plt.rcParams.update({
        'font.size': 28,                  # Global base font size
        'axes.titlesize': 34,             # Bold title sizes
        'axes.labelsize': 32,             # Axis labels
        'xtick.labelsize': 26,            # X axis tick labels
        'ytick.labelsize': 26,            # Y axis tick labels
        'legend.fontsize': 22,            # Legend elements
        'figure.titlesize': 38,           # Figure title scale
        'axes.edgecolor': '#222222',     # Solid contrast border definition
        'axes.linewidth': 2.0,            # Thicker frame boundaries
        'xtick.major.size': 10,           # Highly visible tick markers
        'xtick.major.width': 2.0,
        'ytick.major.size': 10,
        'ytick.major.width': 2.0
    })

    # Expanded canvas size slightly to allow spacious padding under aggressive font scaling
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
    
    for i, judge in enumerate(judges):
        ax1 = axes[i]
        ax2 = ax1.twinx() 
        
        for app, color, label, marker in zip(approaches, ["#1f77b4", "#d62728"], 
                                             ["Persona", "Transformation"], ["o", "s"]):
            
            data = all_stats[app].get(judge, {})
            depths = sorted([int(d) for d in data.keys()])
            
            if not depths:
                continue

            asr_values = [data[str(d)]['asr'] for d in depths]
            sp_values  = [data[str(d)]['preservation'] for d in depths]
            
            linestyle = "-" if app == "persona" else "--"
            
            # Left Axis (ASR) - Scaled stroke weight and marker radius
            ax1.plot(depths, asr_values, label=f"{label} ASR", color=color, 
                     marker=marker, linestyle=linestyle, linewidth=4, markersize=12, alpha=0.35)
            
            # Right Axis (SP) - Heavily reinforced curves to stand out as primary metric
            ax2.plot(depths, sp_values, label=f"{label} SP", color=color, 
                     marker=marker, linestyle=linestyle, linewidth=6.0, markersize=12)

        # Layout, spacing limits, and grid parameters
        ax1.set_title(f"{judge}", fontweight='bold', pad=20)
        ax1.set_xlabel("Mutation Depth ($d$)", labelpad=15)
        ax1.set_xticks(depths)
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Absolute clipping safety margins
        ax1.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # Sparing left labels to maximize plot dimensions
        if i == 0:
            ax1.set_ylabel("Attack Success Rate (ASR)", labelpad=15)
        else:
            ax1.set_yticklabels([])
        
        # Shared styling definitions
        if i == 2:
            ax2.set_ylabel("Semantic Preservation (SP)", labelpad=15)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            combined_lines = lines1[:2] + lines2[:2]
            combined_labels = labels1[:2] + labels2[:2]
            ax1.legend(combined_lines, combined_labels, loc='lower left', framealpha=0.95, facecolor='white')
        else:
            ax2.set_yticklabels([])

    plt.tight_layout()
    
    # Asset directory initialization
    output_dir = Path("combined_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "asr_sp_recovery_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Combined academic double-column plot saved to {output_file}")

if __name__ == "__main__":
    main()