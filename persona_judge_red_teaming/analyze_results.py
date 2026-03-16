"""
analyze_results.py — analysis script for the persona-based red-teaming approach.

Two modes controlled by --cluster flag:
  --cluster   Run HDBSCAN clustering + UMAP on personas, then plot per-cluster metrics.
  (default)   Skip clustering; compute and plot all metrics treating each persona
              as an individual operator (scales to 20k+ personas).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from metrics import (
    EmbeddingCache,
    MutationEdge,
    attack_success_rate_per_operator,
    collect_results,
    cumulative_metrics_by_depth,
    depth_of_success_per_operator,
    drift_rate_per_operator,
    group_edges_by_operator,
    group_edges_by_root,
    jco_per_operator,
    mei_per_operator,
    root_level_asr_per_operator,
    root_level_depth_of_success,
    root_level_depth_of_success_per_operator,
    root_level_jco_per_operator,
    root_level_semantic_preservation,
    root_level_semantic_preservation_per_operator,
    save_results_json,
    semantic_preservation,
    semantic_preservation_per_operator,
)
from persona_clustering import (
    FIELD_WEIGHTS,
    cluster_personas_hdbscan,
    generate_persona_umap,
    plot_persona_umap,
    relabel_edges_with_clusters,
    save_cluster_artifacts,
)
from plots import (
    build_color_map,
    plot_asr,
    plot_asr_sp_heatmap,
    plot_asr_vs_sp,
    plot_cumulative_metrics_vs_depth,
    plot_depth_of_success,
    plot_depth_sorted,
    plot_drift_rate,
    plot_drift_sorted,
    plot_jco,
    plot_mei,
    plot_root_depth_sorted,
    plot_semantic_preservation,
    plot_semantic_preservation_violin,
    plot_sp_sorted,
    plot_success_vs_sp,
)


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def load_persona_edges_from_archive(
    archive_path: str,
    *,
    strict: bool = False,
) -> List[MutationEdge]:
    with open(archive_path, "r") as f:
        archive = json.load(f)

    edges: List[MutationEdge] = []
    n_iter = len(archive)
    print(f"[LOAD] {archive_path} — {n_iter} iterations")

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

        for attempt in iter_data.get("attempts", []):
            try:
                refinement_iter = int(attempt.get("refinement_iter", 0))
                parent_prompt   = attempt["parent_prompt"]
                parent_fitness  = float(attempt["parent_score"])

                persona = attempt.get("persona", {})
                op_meta = {
                    "persona_name":                 persona.get("name",                 "unknown"),
                    "persona_style":                persona.get("style",                ""),
                    "persona_tone":                 persona.get("tone",                 ""),
                    "persona_domain":               persona.get("domain",               ""),
                    "persona_obfuscation_strategy": persona.get("obfuscation_strategy", ""),
                }

                mutated   = attempt.get("mutated_prompts", [])
                fitness   = attempt.get("fitness", [])
                delta     = attempt.get("delta_fitness", [])
                successes = attempt.get("success", [])

                n = min(len(mutated), len(fitness), len(delta), len(successes))
                if n == 0:
                    continue

                for i in range(n):
                    edges.append(MutationEdge(
                        iteration_id      = iteration_id,
                        refinement_iter   = refinement_iter,
                        root_prompt       = root_prompt,
                        root_fitness      = root_fitness,
                        source_dataset    = source,
                        parent_prompt     = parent_prompt,
                        parent_fitness    = parent_fitness,
                        child_prompt      = mutated[i],
                        child_fitness     = float(fitness[i]),
                        delta_fitness     = float(delta[i]),
                        success           = bool(successes[i]),
                        operator_type     = "persona",
                        operator_name     = op_meta["persona_name"],
                        operator_metadata = op_meta,
                        is_root_edge      = (parent_prompt == root_prompt),
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
        print(f"[LOAD] Loading edges from: {path}")
        all_edges.extend(load_persona_edges_from_archive(str(path), strict=strict))

    unique_personas = len({e.operator_name for e in all_edges})
    print(f"[LOAD] {len(all_edges)} edges, {unique_personas} unique personas "
          f"from {len(paths)} archive(s).")
    return all_edges


# ---------------------------------------------------------------------------
# Per-persona sorted helpers (for large-scale, non-clustered analysis)
# ---------------------------------------------------------------------------

def _sp_sorted_per_persona(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    level: str = "edge",
) -> List[tuple]:
    """Returns [(persona_name, sp_value), ...] sorted descending."""
    if level == "edge":
        data = [
            (e.operator_metadata.get("persona_name", "unknown"),
             embedder.cosine(e.root_prompt, e.child_prompt))
            for e in edges
        ]
    else:
        by_root = group_edges_by_root(edges)
        data = []
        for root, es in by_root.items():
            max_depth = max(e.refinement_iter for e in es)
            terminals = [e for e in es if e.success or e.refinement_iter == max_depth]
            for e in terminals:
                name = e.operator_metadata.get("persona_name", "unknown")
                sim  = embedder.cosine(e.root_prompt, e.child_prompt)
                data.append((name, sim))
    return sorted(data, key=lambda x: x[1], reverse=True)


def _depth_sorted_per_persona(edges: List[MutationEdge]) -> List[tuple]:
    return sorted(
        [(e.operator_metadata.get("persona_name", "unknown"),
          e.refinement_iter if e.success else 0)
         for e in edges],
        key=lambda x: x[1], reverse=True,
    )


def _root_depth_sorted(edges: List[MutationEdge], *, label: str = "min") -> List[tuple]:
    by_root = group_edges_by_root(edges)
    result  = []
    for root, es in by_root.items():
        depths = [e.refinement_iter for e in es if e.success]
        if depths:
            val = min(depths) if label == "min" else float(np.mean(depths))
            result.append((root, val))
    return result


def _drift_sorted_per_persona(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
) -> List[tuple]:
    from metrics import compute_semantic_drift_rate
    data = [
        (e.operator_metadata.get("persona_name", "unknown"),
         compute_semantic_drift_rate([e], embedder)["mean"])
        for e in edges
    ]
    return sorted(data, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main_analysis(
    archive_path: Union[str, List[str]],
    vis_dir: str,
    *,
    do_cluster: bool = False,
) -> None:
    edges_raw = load_edges(archive_path)
    embedder  = EmbeddingCache(device="cuda:0")

    if do_cluster:
        _run_clustered_analysis(edges_raw, embedder, vis_dir)
    else:
        _run_flat_analysis(edges_raw, embedder, vis_dir)


def _run_clustered_analysis(
    edges_raw: List[MutationEdge],
    embedder: EmbeddingCache,
    vis_dir: str,
) -> None:
    """Cluster personas with HDBSCAN, then run per-cluster metric plots."""
    clustering = cluster_personas_hdbscan(
        edges_raw, embedder,
        field_weights=FIELD_WEIGHTS,
        vis_dir=vis_dir,
        min_cluster_size=40,
        min_samples=40,
    )
    print("[ANALYSIS] Clustering complete.")

    save_cluster_artifacts(
        clustering["cluster_members"],
        clustering["cluster_summaries"],
        vis_dir,
    )

    # UMAP
    umap_df = generate_persona_umap(
        edges_raw, embedder,
        clustering["cluster_summaries"],
        weights=FIELD_WEIGHTS,
    )
    plot_persona_umap(umap_df, vis_dir)
    print("[ANALYSIS] UMAP done.")

    edges   = relabel_edges_with_clusters(edges_raw, clustering["persona_to_cluster"])
    grouped = group_edges_by_operator(edges)
    op_names  = list(grouped.keys())
    color_map = build_color_map(op_names, palette="tab10")

    print(f"[ANALYSIS] {len(op_names)} clusters: {op_names}")

    # All standard per-operator plots
    _run_per_operator_plots(edges, grouped, embedder, vis_dir, color_map)

    results = collect_results(edges, grouped, embedder)
    save_results_json(results, os.path.join(vis_dir, "persona_attack_metrics.json"))
    print("[ANALYSIS] Metrics JSON saved.")


def _run_flat_analysis(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    vis_dir: str,
) -> None:
    """
    No clustering — treat each persona as its own operator.
    For very large persona sets, uses sorted-bar distribution plots
    instead of named-bar charts (which would be unreadable at 20k+ labels).
    """
    print("[ANALYSIS] Running flat (no-cluster) persona analysis.")

    # Pre-warm embedding cache with all unique texts in a single batched call.
    # Without this, embed() is called ~15k+ times individually in the metric
    # loops, each invoking model.encode() on a single string — extremely slow.
    all_texts = list({
        t for e in edges
        for t in (e.root_prompt, e.parent_prompt, e.child_prompt)
    })
    embedder.batch_embed(all_texts)
    print(f"[ANALYSIS] Embedding cache warmed with {len(all_texts)} unique texts.", flush=True)

    # ---- Sorted SP distributions
    edge_sp = _sp_sorted_per_persona(edges, embedder, level="edge")
    plot_sp_sorted(edge_sp, vis_dir, level="edge", color="#3498db")
    print("[ANALYSIS] SP_G sorted done.")

    root_sp = _sp_sorted_per_persona(edges, embedder, level="root")
    plot_sp_sorted(root_sp, vis_dir, level="root", color="#2d39bd")
    print("[ANALYSIS] SP_R sorted done.")

    # ---- Sorted depth distributions
    edge_depth = _depth_sorted_per_persona(edges)
    plot_depth_sorted(edge_depth, vis_dir, color="#079D68")
    print("[ANALYSIS] Edge depth sorted done.")

    root_depth_min = _root_depth_sorted(edges, label="min")
    plot_root_depth_sorted(root_depth_min, vis_dir, label="min", color="#9b59b6")
    print("[ANALYSIS] Root depth (min) sorted done.")

    root_depth_avg = _root_depth_sorted(edges, label="avg")
    plot_root_depth_sorted(root_depth_avg, vis_dir, label="avg", color="#773a8f")
    print("[ANALYSIS] Root depth (avg) sorted done.")

    # ---- Sorted drift distribution
    drift_data = _drift_sorted_per_persona(edges, embedder)
    plot_drift_sorted(drift_data, vis_dir, color="#e74c3c")
    print("[ANALYSIS] Drift sorted done.")

    # ---- Cumulative metrics vs depth
    cum_stats = cumulative_metrics_by_depth(edges, embedder)
    plot_cumulative_metrics_vs_depth(cum_stats, vis_dir)
    print("[ANALYSIS] Cumulative metrics vs depth done.")

    # ---- Success vs SP scatter
    plot_success_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Success vs SP scatter done.")

    # ---- Global metrics JSON (no per-operator breakdown)
    results = collect_results(edges, grouped=None, embedder=embedder)
    save_results_json(results, os.path.join(vis_dir, "persona_attack_metrics.json"))
    print("[ANALYSIS] Metrics JSON saved.")


def _run_per_operator_plots(
    edges: List[MutationEdge],
    grouped: dict,
    embedder: EmbeddingCache,
    vis_dir: str,
    color_map: dict,
) -> None:
    """Standard named-bar plots — used after clustering when operator count is small."""

    # ASR
    asr_per      = attack_success_rate_per_operator(grouped)
    root_asr_per = root_level_asr_per_operator(grouped)
    plot_asr(asr_per,      vis_dir, color_map, level="edge")
    plot_asr(root_asr_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] ASR done.")

    # SP
    sp_per      = semantic_preservation_per_operator(grouped, embedder)
    root_sp_per = root_level_semantic_preservation_per_operator(grouped, embedder)
    plot_semantic_preservation(sp_per,      vis_dir, color_map, level="edge")
    plot_semantic_preservation(root_sp_per, vis_dir, color_map, level="root")
    plot_semantic_preservation_violin(grouped, embedder, vis_dir, color_map)
    print("[ANALYSIS] SP done.")

    # Heatmaps + scatter
    plot_asr_sp_heatmap(asr_per,      sp_per,      vis_dir, level="edge")
    plot_asr_sp_heatmap(root_asr_per, root_sp_per, vis_dir, level="root")
    plot_asr_vs_sp(asr_per,      sp_per,      vis_dir, color_map, level="edge")
    plot_asr_vs_sp(root_asr_per, root_sp_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] ASR vs SP done.")

    # JCO
    jco_per      = jco_per_operator(grouped, embedder)
    root_jco_per = root_level_jco_per_operator(grouped, embedder)
    plot_jco(jco_per,      vis_dir, color_map, level="edge")
    plot_jco(root_jco_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] JCO done.")

    # MEI
    mei_per = mei_per_operator(grouped, embedder)
    plot_mei(mei_per, vis_dir, color_map)
    print("[ANALYSIS] MEI done.")

    # Depth
    depth_per      = depth_of_success_per_operator(grouped)
    root_depth_per = root_level_depth_of_success_per_operator(grouped)
    plot_depth_of_success(depth_per,      vis_dir, color_map, level="edge")
    plot_depth_of_success(root_depth_per, vis_dir, color_map, level="root")
    print("[ANALYSIS] Depth done.")

    # Cumulative metrics
    cum_stats = cumulative_metrics_by_depth(edges, embedder)
    plot_cumulative_metrics_vs_depth(cum_stats, vis_dir)
    print("[ANALYSIS] Cumulative metrics vs depth done.")

    # Drift
    drift_per = drift_rate_per_operator(grouped, embedder)
    plot_drift_rate(drift_per, vis_dir, color_map)
    print("[ANALYSIS] Drift done.")

    # Success vs SP scatter
    plot_success_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Success vs SP scatter done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persona approach — results analysis")
    parser.add_argument("--results_archive", type=str, default=None,
                        help="Path to a single red-teaming JSON archive")
    parser.add_argument("--results_folders", type=str, nargs="+", default=None,
                        help="One or more folders each containing red_teaming_archive.json")
    parser.add_argument("--cluster", action="store_true",
                        help="Run HDBSCAN clustering + UMAP before plotting")
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
    main_analysis(archive_path, vis_dir, do_cluster=args.cluster)
    exit(0)