"""
analyze_results_destylization.py — Analysis script for destylized archives,
transformation-based red-teaming approach.

Reads one or more red_teaming_archive_destylized.json files and computes
metrics that quantify the de-stylization experiment:

    - Recovery Rate: fraction of judge-bypassing mutations re-caught after
      destylization (safe→harmful). High RR = attacks were purely stylistic.
    - DES: delta in judge score after destylization. Positive = harmful content
      was surfaced; zero/negative = semantic drift was genuine.
    - SP destylized vs root: how close did destylization bring the prompt back
      to the original root? High = destylizer reversed the obfuscation well.
    - SP destylized vs mutated: how much did destylization change the surface
      form? Low = significant style was removed.
    - Recovery by depth: does recovery rate decrease at higher refinement depth?
    - Recovery by operator: which transformation types are most/least recoverable?
    - Cumulative recovery vs depth: cumulative view of the above.

Usage:
    python analyze_results_destylization.py \\
        --results_folders results/LlamaGuard/20260309_141041 ... \\
        --vis_dir visualizations_destylization \\
        [--gpu]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from metrics import EmbeddingCache
from metrics_destylization import (
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
from plots import build_color_map
from plots_destylization import (
    plot_cumulative_recovery,
    plot_des,
    plot_des_by_depth,
    plot_recovery_rate,
    plot_recovery_rate_by_depth,
    plot_recovery_vs_sp,
    plot_sp_destylized,
)


def build_color_map_local(operators: List[str]) -> dict:
    try:
        return build_color_map(operators, palette="tab20")
    except Exception:
        import matplotlib.pyplot as plt
        cmap   = plt.get_cmap("tab20")
        colors = [cmap(i / max(len(operators), 1)) for i in range(len(operators))]
        return {op: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                for op, c in zip(operators, colors)}


def main_analysis(archive_paths: List[str], vis_dir: str, *, use_gpu: bool) -> None:
    device  = "cuda:0" if use_gpu else "cpu"
    edges   = load_destylized_edges(archive_paths, approach="transformation")
    embedder = EmbeddingCache(device=device)

    if not edges:
        print("[ANALYSIS] No destylized edges found. Exiting.", flush=True)
        return

    # Pre-warm embedding cache
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
    """
    grouped   = group_destylized_by_operator(edges)
    op_names  = list(grouped.keys())
    color_map = build_color_map_local(op_names)

    print(f"[ANALYSIS] {len(edges)} edges, {len(op_names)} operators.", flush=True)

    # --- Recovery Rate
    rr_per_op    = recovery_rate_per_operator(grouped)
    rr_per_depth = recovery_rate_per_depth(edges)
    plot_recovery_rate(rr_per_op, vis_dir, color_map)
    plot_recovery_rate_by_depth(rr_per_depth, vis_dir)
    print("[ANALYSIS] Recovery Rate done.")

    # --- DES
    des_op    = des_per_operator(grouped)
    des_depth = des_per_depth(edges)
    plot_des(des_op, vis_dir, color_map)
    plot_des_by_depth(des_depth, vis_dir)
    print("[ANALYSIS] DES done.")

    # --- SP destylized vs root
    sp_root = sp_destylized_vs_root_per_operator(grouped, embedder)
    plot_sp_destylized(sp_root, vis_dir, color_map, vs="root")
    print("[ANALYSIS] SP destylized vs root done.")

    # --- SP destylized vs mutated
    sp_mut = sp_destylized_vs_mutated_per_operator(grouped, embedder)
    plot_sp_destylized(sp_mut, vis_dir, color_map, vs="mutated")
    print("[ANALYSIS] SP destylized vs mutated done.")

    # --- Cumulative recovery
    cum = cumulative_recovery_by_depth(edges)
    plot_cumulative_recovery(cum, vis_dir)
    print("[ANALYSIS] Cumulative recovery done.")

    # --- Recovery vs SP scatter
    plot_recovery_vs_sp(edges, embedder, vis_dir)
    print("[ANALYSIS] Recovery vs SP scatter done.")
    """
    # --- Save metrics JSON
    #results = collect_destylized_results(edges, grouped, embedder)
    results = collect_destylized_results(edges=edges, grouped=None, embedder=embedder)
    save_destylized_results_json(
        results,
        os.path.join(vis_dir, "destylized_attack_metrics.json"),
    )
    print("[ANALYSIS] Metrics JSON saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Destylized archive analysis — transformation approach"
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
    main_analysis(archive_paths, vis_dir, use_gpu=args.gpu)