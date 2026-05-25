"""
analyze_results_destylization.py — Analysis script for destylized archives,
persona-based red-teaming approach.

Two modes controlled by --cluster flag:
  --cluster   Run HDBSCAN clustering on personas first, then per-cluster plots.
  (default)   Flat mode: sorted-bar distribution plots, no per-operator breakdown.

Usage:
    python analyze_results_destylization.py \\
        --results_folders results/LlamaGuard/20260309_142402 ... \\
        --vis_dir visualizations_destylization \\
        [--cluster] \\
        [--gpu]
"""
from __future__ import annotations

import argparse
import os
from typing import List

from metrics import EmbeddingCache, MutationEdge
from metrics_destylization import (
    DestylizedEdge,
    collect_destylized_results,
    cumulative_recovery_by_depth,
    des_per_depth,
    des_per_operator,
    group_destylized_by_operator,
    load_destylized_edges,
    recovery_rate_per_depth,
    recovery_rate_per_operator,
    save_destylized_results_json,
    sp_destylized_vs_mutated_per_operator,
    sp_destylized_vs_root_per_operator,
)
from persona_clustering import (
    FIELD_WEIGHTS,
    cluster_personas_hdbscan,
    generate_persona_umap,
    plot_persona_umap,
    relabel_edges_with_clusters,
    save_cluster_artifacts,
)
from plots import build_color_map
from plots_destylization import (
    plot_cumulative_recovery,
    plot_des,
    plot_des_by_depth,
    plot_des_sorted,
    plot_recovery_rate,
    plot_recovery_rate_by_depth,
    plot_recovery_rate_sorted,
    plot_recovery_vs_sp,
    plot_sp_destylized,
    plot_sp_destylized_sorted,
)


# ---------------------------------------------------------------------------
# Cluster bridge
# ---------------------------------------------------------------------------

def _destylized_to_mutation_edges(edges: List[DestylizedEdge]) -> List[MutationEdge]:
    """
    Convert DestylizedEdge list to MutationEdge list for use with
    persona_clustering functions which expect MutationEdge.
    """
    from metrics import MutationEdge
    return [
        MutationEdge(
            iteration_id      = e.iteration_id,
            refinement_iter   = e.refinement_iter,
            root_prompt       = e.root_prompt,
            root_fitness      = e.root_fitness,
            source_dataset    = e.source_dataset,
            parent_prompt     = e.parent_prompt,
            parent_fitness    = e.parent_fitness,
            child_prompt      = e.child_prompt,
            child_fitness     = e.child_fitness,
            delta_fitness     = e.delta_fitness,
            success           = e.success,
            operator_type     = e.operator_type,
            operator_name     = e.operator_name,
            operator_metadata = e.operator_metadata,
            is_root_edge      = e.is_root_edge,
        )
        for e in edges
    ]


def _relabel_destylized(
    edges: List[DestylizedEdge],
    persona_to_cluster: dict,
) -> List[DestylizedEdge]:
    """Re-label operator_name with cluster id, preserving all other fields."""
    from dataclasses import replace
    relabeled = []
    for e in edges:
        cluster = persona_to_cluster.get(e.operator_name, "noise")
        relabeled.append(
            DestylizedEdge(
                iteration_id      = e.iteration_id,
                refinement_iter   = e.refinement_iter,
                root_prompt       = e.root_prompt,
                root_fitness      = e.root_fitness,
                source_dataset    = e.source_dataset,
                parent_prompt     = e.parent_prompt,
                parent_fitness    = e.parent_fitness,
                child_prompt      = e.child_prompt,
                child_fitness     = e.child_fitness,
                delta_fitness     = e.delta_fitness,
                success           = e.success,
                operator_type     = e.operator_type,
                operator_name     = str(cluster),
                operator_metadata = e.operator_metadata,
                is_root_edge      = e.is_root_edge,
                destylized_prompt  = e.destylized_prompt,
                destylized_fitness = e.destylized_fitness,
                destylized_success = e.destylized_success,
                recovered          = e.recovered,
            )
        )
    return relabeled


# ---------------------------------------------------------------------------
# Analysis modes
# ---------------------------------------------------------------------------

def _warm_cache(edges: List[DestylizedEdge], embedder: EmbeddingCache) -> None:
    all_texts = list({
        t for e in edges
        for t in (e.root_prompt, e.child_prompt)
        if t is not None
    } | {
        e.destylized_prompt for e in edges
        if e.destylized_prompt is not None
    })
    embedder.batch_embed(all_texts)
    print(f"[ANALYSIS] Embedding cache warmed with {len(all_texts)} texts.", flush=True)


def _run_per_operator_plots(
    edges: List[DestylizedEdge],
    grouped: dict,
    embedder: EmbeddingCache,
    vis_dir: str,
    color_map: dict,
) -> None:
    rr_per_op = recovery_rate_per_operator(grouped)
    plot_recovery_rate(rr_per_op, vis_dir, color_map)
    print("[ANALYSIS] Recovery Rate per operator done.")

    des_op = des_per_operator(grouped)
    plot_des(des_op, vis_dir, color_map)
    print("[ANALYSIS] DES per operator done.")

    sp_root = sp_destylized_vs_root_per_operator(grouped, embedder)
    plot_sp_destylized(sp_root, vis_dir, color_map, vs="root")
    print("[ANALYSIS] SP destylized vs root done.")

    sp_mut = sp_destylized_vs_mutated_per_operator(grouped, embedder)
    plot_sp_destylized(sp_mut, vis_dir, color_map, vs="mutated")
    print("[ANALYSIS] SP destylized vs mutated done.")

    rr_depth = recovery_rate_per_depth(edges)
    plot_recovery_rate_by_depth(rr_depth, vis_dir)
    print("[ANALYSIS] Recovery Rate by depth done.")

    des_depth = des_per_depth(edges)
    plot_des_by_depth(des_depth, vis_dir)
    print("[ANALYSIS] DES by depth done.")

    cum = cumulative_recovery_by_depth(edges)
    plot_cumulative_recovery(cum, vis_dir)
    print("[ANALYSIS] Cumulative recovery done.")

    plot_recovery_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Recovery vs SP scatter done.")


def _run_clustered_analysis(
    edges_raw: List[DestylizedEdge],
    embedder: EmbeddingCache,
    vis_dir: str,
) -> None:
    # Cluster using MutationEdge interface (persona_clustering expects it)
    mutation_edges = _destylized_to_mutation_edges(edges_raw)
    clustering = cluster_personas_hdbscan(
        mutation_edges, embedder,
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

    umap_df = generate_persona_umap(
        mutation_edges, embedder,
        clustering["cluster_summaries"],
        weights=FIELD_WEIGHTS,
    )
    plot_persona_umap(umap_df, vis_dir)
    print("[ANALYSIS] UMAP done.")

    edges     = _relabel_destylized(edges_raw, clustering["persona_to_cluster"])
    grouped   = group_destylized_by_operator(edges)
    op_names  = list(grouped.keys())
    color_map = build_color_map(op_names, palette="tab10")

    print(f"[ANALYSIS] {len(op_names)} clusters: {op_names}")
    _run_per_operator_plots(edges, grouped, embedder, vis_dir, color_map)

    results = collect_destylized_results(edges, grouped, embedder)
    save_destylized_results_json(
        results,
        os.path.join(vis_dir, "destylized_persona_metrics.json"),
    )
    print("[ANALYSIS] Metrics JSON saved.")


def _run_flat_analysis(
    edges: List[DestylizedEdge],
    embedder: EmbeddingCache,
    vis_dir: str,
) -> None:
    print("[ANALYSIS] Running flat (no-cluster) destylized persona analysis.")

    grouped = group_destylized_by_operator(edges)

    # Sorted distribution plots (readable at 20k+ personas)
    rr_per_op = recovery_rate_per_operator(grouped)
    plot_recovery_rate_sorted(rr_per_op, vis_dir)
    print("[ANALYSIS] Recovery Rate sorted done.")

    des_op = des_per_operator(grouped)
    plot_des_sorted(des_op, vis_dir)
    print("[ANALYSIS] DES sorted done.")

    sp_root = sp_destylized_vs_root_per_operator(grouped, embedder)
    plot_sp_destylized_sorted(sp_root, vis_dir, vs="root")
    print("[ANALYSIS] SP destylized vs root sorted done.")

    sp_mut = sp_destylized_vs_mutated_per_operator(grouped, embedder)
    plot_sp_destylized_sorted(sp_mut, vis_dir, vs="mutated")
    print("[ANALYSIS] SP destylized vs mutated sorted done.")

    # Depth-based plots (not per-operator, global)
    rr_depth = recovery_rate_per_depth(edges)
    plot_recovery_rate_by_depth(rr_depth, vis_dir)
    print("[ANALYSIS] Recovery Rate by depth done.")

    des_depth = des_per_depth(edges)
    plot_des_by_depth(des_depth, vis_dir)
    print("[ANALYSIS] DES by depth done.")

    cum = cumulative_recovery_by_depth(edges)
    plot_cumulative_recovery(cum, vis_dir)
    print("[ANALYSIS] Cumulative recovery done.")

    plot_recovery_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Recovery vs SP scatter done.")

    results = collect_destylized_results(edges, grouped=None, embedder=embedder)
    save_destylized_results_json(
        results,
        os.path.join(vis_dir, "destylized_persona_metrics.json"),
    )
    print("[ANALYSIS] Metrics JSON saved.")


def main_analysis(
    archive_paths: List[str],
    vis_dir: str,
    *,
    do_cluster: bool,
    use_gpu: bool,
) -> None:
    device   = "cuda:0" if use_gpu else "cpu"
    edges    = load_destylized_edges(archive_paths, approach="persona")
    embedder = EmbeddingCache(device=device)

    if not edges:
        print("[ANALYSIS] No destylized edges found. Exiting.", flush=True)
        return

    _warm_cache(edges, embedder)

    if do_cluster:
        _run_clustered_analysis(edges, embedder, vis_dir)
    else:
        _run_flat_analysis(edges, embedder, vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Destylized archive analysis — persona approach"
    )
    parser.add_argument(
        "--results_archive", type=str, default=None,
        help="Path to a single red_teaming_archive_destylized.json",
    )
    parser.add_argument(
        "--results_folders", type=str, nargs="+", default=None,
        help="One or more folders each containing red_teaming_archive_destylized.json",
    )
    parser.add_argument(
        "--vis_dir", type=str, default="visualizations_destylized",
        help="Directory to save visualizations and metrics JSON",
    )
    parser.add_argument(
        "--cluster", action="store_true",
        help="Run HDBSCAN clustering + UMAP before plotting",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU for sentence-transformer embeddings",
    )
    args = parser.parse_args()

    if args.results_archive is not None:
        archive_paths = [os.path.abspath(args.results_archive)]
    else:
        archive_paths = []
        for folder in args.results_folders:
            path = os.path.abspath(
                os.path.join(folder, "red_teaming_archive_destylized_qwen3.5.json")
            )
            if not os.path.isfile(path):
                print(f"[WARNING] No destylized archive at: {path}")
            else:
                archive_paths.append(path)

    vis_dir = os.path.abspath(args.vis_dir)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[ANALYSIS] Archives: {archive_paths}")
    print(f"[ANALYSIS] Output:   {vis_dir}")
    main_analysis(archive_paths, vis_dir, do_cluster=args.cluster, use_gpu=args.gpu)