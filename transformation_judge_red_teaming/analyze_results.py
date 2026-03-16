"""
analyze_results.py — analysis script for the transformation-based red-teaming approach.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Union

from metrics import (
    EmbeddingCache,
    MutationEdge,
    attack_success_rate_per_operator,
    collect_results,
    cumulative_metrics_by_depth,
    depth_of_success_per_operator,
    drift_rate_per_operator,
    group_edges_by_operator,
    jco_per_operator,
    mei_per_operator,
    root_level_asr_per_operator,
    root_level_depth_of_success_per_operator,
    root_level_jco_per_operator,
    root_level_semantic_preservation_per_operator,
    save_results_json,
    semantic_preservation_per_operator,
)
from plots import (
    build_color_map,
    plot_asr,
    plot_asr_sp_heatmap,
    plot_asr_vs_sp,
    plot_cumulative_metrics_vs_depth,
    plot_depth_of_success,
    plot_drift_rate,
    plot_jco,
    plot_mei,
    plot_semantic_preservation,
    plot_semantic_preservation_violin,
    plot_success_vs_sp,
)


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def load_mutation_edges_from_archive(
    archive_path: str,
    *,
    strict: bool = False,
) -> List[MutationEdge]:
    try:
        with open(archive_path, "r") as f:
            archive = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load archive from {archive_path}: {e}")

    edges: List[MutationEdge] = []

    for iter_key, iter_data in archive.items():
        if iter_key.startswith("_"):
            continue  # skip metadata keys like _metadata
        try:
            iteration_id = int(iter_key)
        except ValueError:
            if strict:
                raise
            continue

        root_prompt  = iter_data.get("parent_prompt")
        root_fitness = float(iter_data.get("parent_fitness", 0.0))
        source       = iter_data.get("parent_metadata", {}).get("source")

        op_info   = iter_data.get("transformation", {})
        op_name   = op_info.get("name", "unknown")
        op_meta   = dict(op_info)

        for attempt in iter_data.get("attempts", []):
            try:
                refinement_iter = int(attempt["refinement_iter"])
                parent_prompt   = attempt["parent_prompt"]
                parent_fitness  = float(attempt["parent_score"])

                mutated   = attempt.get("mutated_prompts", [])
                fitness   = attempt.get("fitness", [])
                delta     = attempt.get("delta_fitness", [])
                successes = attempt.get("success", [])

                n = min(len(mutated), len(fitness), len(delta), len(successes))
                if n == 0:
                    continue

                for i in range(n):
                    edges.append(MutationEdge(
                        iteration_id    = iteration_id,
                        refinement_iter = refinement_iter,
                        root_prompt     = root_prompt,
                        root_fitness    = root_fitness,
                        source_dataset  = source,
                        parent_prompt   = parent_prompt,
                        parent_fitness  = parent_fitness,
                        child_prompt    = mutated[i],
                        child_fitness   = float(fitness[i]),
                        delta_fitness   = float(delta[i]),
                        success         = bool(successes[i]),
                        operator_type   = "transformation",
                        operator_name   = op_name,
                        operator_metadata = op_meta,
                        is_root_edge    = (parent_prompt == root_prompt),
                    ))
            except Exception:
                if strict:
                    raise
                continue

    return edges


def load_edges(
    archive_paths: Union[str, Iterable[str]],
    *,
    strict: bool = False,
) -> List[MutationEdge]:
    if isinstance(archive_paths, str):
        p = Path(archive_paths)
        paths = sorted(p.glob("*.json")) if p.is_dir() else sorted(Path().glob(archive_paths))
    else:
        paths = [Path(p) for p in archive_paths]

    all_edges: List[MutationEdge] = []
    for path in paths:
        all_edges.extend(load_mutation_edges_from_archive(str(path), strict=strict))
    print(f"[LOAD] {len(all_edges)} edges from {len(paths)} archive(s).")
    return all_edges


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main_analysis(archive_path: Union[str, List[str]], vis_dir: str) -> None:
    edges   = load_edges(archive_path)
    grouped = group_edges_by_operator(edges)
    embedder = EmbeddingCache(device="cuda:0")

    op_names  = list(grouped.keys())
    color_map = build_color_map(op_names, palette="Paired")

    print(f"[ANALYSIS] {len(edges)} edges, {len(op_names)} operators, "
          f"{len({e.root_prompt for e in edges})} unique roots.")

    # Pre-warm embedding cache with all unique texts in a single batched call.
    # Without this, embed() is called ~15k+ times individually in the metric
    # loops, each invoking model.encode() on a single string — extremely slow.
    all_texts = list({
        t for e in edges
        for t in (e.root_prompt, e.parent_prompt, e.child_prompt)
    })
    embedder.batch_embed(all_texts)
    print(f"[ANALYSIS] Embedding cache warmed with {len(all_texts)} unique texts.", flush=True)

    # ---- Edge-level ASR
    asr_per = attack_success_rate_per_operator(grouped)
    plot_asr(asr_per, vis_dir, color_map, level="edge")
    print("[ANALYSIS] ASR_G done.")

    # ---- Root-level ASR
    root_asr_per = root_level_asr_per_operator(grouped)
    plot_asr(root_asr_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] ASR_R done.")

    # ---- Edge-level SP
    sp_per = semantic_preservation_per_operator(grouped, embedder)
    plot_semantic_preservation(sp_per, vis_dir, color_map, level="edge")
    plot_semantic_preservation_violin(grouped, embedder, vis_dir, color_map)
    print("[ANALYSIS] SP_G done.")

    # ---- Root-level SP
    root_sp_per = root_level_semantic_preservation_per_operator(grouped, embedder)
    plot_semantic_preservation(root_sp_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] SP_R done.")

    # ---- ASR vs SP heatmap
    plot_asr_sp_heatmap(asr_per,      sp_per,      vis_dir, level="edge")
    plot_asr_sp_heatmap(root_asr_per, root_sp_per, vis_dir, level="root")
    print("[ANALYSIS] ASR vs SP heatmap done.")

    # ---- ASR vs SP scatter
    plot_asr_vs_sp(asr_per,      sp_per,      vis_dir, color_map, level="edge")
    plot_asr_vs_sp(root_asr_per, root_sp_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] ASR vs SP scatter done.")

    # ---- Edge-level JCO
    jco_per = jco_per_operator(grouped, embedder)
    plot_jco(jco_per, vis_dir, color_map, level="edge")
    print("[ANALYSIS] JCO_G done.")

    # ---- Root-level JCO
    root_jco_per = root_level_jco_per_operator(grouped, embedder)
    plot_jco(root_jco_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] JCO_R done.")

    # ---- MEI
    mei_per = mei_per_operator(grouped, embedder)
    plot_mei(mei_per, vis_dir, color_map)
    print("[ANALYSIS] MEI done.")

    # ---- Edge-level Depth of Success
    depth_per = depth_of_success_per_operator(grouped)
    plot_depth_of_success(depth_per, vis_dir, color_map, level="edge")
    print("[ANALYSIS] Depth_G done.")

    # ---- Root-level Depth of Success
    root_depth_per = root_level_depth_of_success_per_operator(grouped)
    plot_depth_of_success(root_depth_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] Depth_R done.")

    # ---- Cumulative metrics vs depth
    cum_stats = cumulative_metrics_by_depth(edges, embedder)
    plot_cumulative_metrics_vs_depth(cum_stats, vis_dir)
    print("[ANALYSIS] Cumulative metrics vs depth done.")

    # ---- Semantic drift
    drift_per = drift_rate_per_operator(grouped, embedder)
    plot_drift_rate(drift_per, vis_dir, color_map)
    print("[ANALYSIS] Semantic drift done.")

    # ---- Success vs SP
    plot_success_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Success vs SP scatter done.")

    # ---- Collect and save all metrics
    results = collect_results(edges, grouped, embedder)
    save_results_json(results, os.path.join(vis_dir, "transformation_attack_metrics.json"))
    print("[ANALYSIS] Metrics JSON saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformation approach — results analysis")
    parser.add_argument("--results_archive",  type=str, default=None,
                        help="Path to a single red-teaming JSON archive")
    parser.add_argument("--results_folders",  type=str, nargs="+", default=None,
                        help="One or more folders each containing red_teaming_archive.json")
    parser.add_argument("--vis_dir", type=str, default="visualizations",
                        help="Directory to save visualizations and metrics JSON")
    args = parser.parse_args()

    if args.results_archive is not None:
        archive_path = os.path.abspath(args.results_archive)
        vis_dir = os.path.abspath(args.vis_dir)
    else:
        archive_path = []
        for folder in args.results_folders:
            try:
                path = os.path.abspath(os.path.join(folder, "red_teaming_archive.json"))
                if not os.path.isfile(path):
                    print(f"[WARNING] No archive found at: {path}")
                else:
                    archive_path.append(path)
            except Exception as e:
                print(f"[ERROR] Failed to process folder: {folder} — {e}")
        vis_dir = os.path.abspath(args.vis_dir)

    os.makedirs(vis_dir, exist_ok=True)
    print(f"[ANALYSIS] Loading from: {archive_path}")
    print(f"[ANALYSIS] Visualizations and metrics will be saved to: {vis_dir}")
    main_analysis(archive_path, vis_dir)
    exit(0)