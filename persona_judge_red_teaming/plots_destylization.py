"""
plots_destylization.py — Visualization functions for destylized archive analysis.

Follows the same style as plots.py: one function per plot, consistent
color scheme, no global state, all functions take vis_dir as output path.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics_destylization import DestylizedEdge, EmbeddingCache

FIGSIZE_BAR  = (12, 5)
FIGSIZE_LINE = (10, 5)
FIGSIZE_WIDE = (14, 5)
DPI          = 150


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}", flush=True)


def _bar(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    color_map: Optional[Dict[str, str]] = None,
    default_color: str = "#2c7bb6",
    ylabel: str = "",
    title:  str = "",
    ylim:   Optional[Tuple] = None,
) -> None:
    colors = [color_map.get(l, default_color) for l in labels] if color_map else [default_color] * len(labels)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7,
        )


# ---------------------------------------------------------------------------
# Recovery Rate plots
# ---------------------------------------------------------------------------

def plot_recovery_rate(
    rr_per_operator: Dict[str, float],
    vis_dir: str,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    ops    = list(rr_per_operator.keys())
    values = [rr_per_operator[op] for op in ops]
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    _bar(ax, ops, values, color_map=color_map,
         default_color="#e74c3c",
         ylabel="Recovery Rate",
         title="Recovery Rate per Operator (safe→harmful after destylization)",
         ylim=(0, 1))
    _save(fig, os.path.join(vis_dir, "recovery_rate_per_operator.png"))


def plot_recovery_rate_by_depth(
    rr_per_depth: Dict[int, float],
    vis_dir: str,
) -> None:
    depths = sorted(rr_per_depth.keys())
    values = [rr_per_depth[d] for d in depths]
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    ax.plot([str(d) for d in depths], values, marker="o", color="#e74c3c", linewidth=2)
    ax.fill_between([str(d) for d in depths], values, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Refinement Depth")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery Rate by Refinement Depth")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, os.path.join(vis_dir, "recovery_rate_by_depth.png"))


def plot_recovery_rate_sorted(
    rr_per_operator: Dict[str, float],
    vis_dir: str,
) -> None:
    """Sorted bar — useful when operator count is large (persona flat mode)."""
    sorted_items = sorted(rr_per_operator.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_items]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(values))
    ax.bar(x, values, color="#e74c3c", edgecolor="none", width=1.0)
    ax.set_xlabel("Operator (sorted by recovery rate)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery Rate Distribution (sorted)")
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, os.path.join(vis_dir, "recovery_rate_sorted.png"))


# ---------------------------------------------------------------------------
# DES plots
# ---------------------------------------------------------------------------

def plot_des(
    des_per_operator: Dict[str, Dict],
    vis_dir: str,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    ops    = list(des_per_operator.keys())
    means  = [des_per_operator[op]["mean"] for op in ops]
    stds   = [des_per_operator[op]["std"]  for op in ops]
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    colors = [color_map.get(op, "#8e44ad") for op in ops] if color_map else ["#8e44ad"] * len(ops)
    ax.bar(ops, means, yerr=stds, color=colors, edgecolor="white",
           linewidth=0.5, capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("DES (destylized_fitness − child_fitness)")
    ax.set_title("Destylization Effectiveness Score per Operator")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, os.path.join(vis_dir, "des_per_operator.png"))


def plot_des_by_depth(
    des_per_depth: Dict[int, Dict],
    vis_dir: str,
) -> None:
    depths = sorted(des_per_depth.keys())
    means  = [des_per_depth[d]["mean"]   for d in depths]
    stds   = [des_per_depth[d]["std"]    for d in depths]
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    ax.plot([str(d) for d in depths], means, marker="o", color="#8e44ad", linewidth=2)
    ax.fill_between(
        [str(d) for d in depths],
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, color="#8e44ad",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Refinement Depth")
    ax.set_ylabel("DES (mean ± std)")
    ax.set_title("Destylization Effectiveness Score by Depth")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, os.path.join(vis_dir, "des_by_depth.png"))


def plot_des_sorted(
    des_per_operator: Dict[str, Dict],
    vis_dir: str,
) -> None:
    sorted_items = sorted(des_per_operator.items(), key=lambda x: x[1]["mean"], reverse=True)
    values = [v["mean"] for _, v in sorted_items]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(values))
    colors = ["#8e44ad" if v >= 0 else "#e74c3c" for v in values]
    ax.bar(x, values, color=colors, edgecolor="none", width=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Operator (sorted by DES)")
    ax.set_ylabel("DES (mean)")
    ax.set_title("DES Distribution (sorted)")
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, os.path.join(vis_dir, "des_sorted.png"))


# ---------------------------------------------------------------------------
# SP plots
# ---------------------------------------------------------------------------

def plot_sp_destylized(
    sp_per_operator: Dict[str, Dict],
    vis_dir: str,
    color_map: Optional[Dict[str, str]] = None,
    *,
    vs: str = "root",  # "root" or "mutated"
) -> None:
    ops    = list(sp_per_operator.keys())
    means  = [sp_per_operator[op]["mean"] for op in ops]
    stds   = [sp_per_operator[op]["std"]  for op in ops]
    label  = "Root Prompt" if vs == "root" else "Mutated Prompt"
    fname  = f"sp_destylized_vs_{vs}_per_operator.png"
    colors = [color_map.get(op, "#27ae60") for op in ops] if color_map else ["#27ae60"] * len(ops)
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.bar(ops, means, yerr=stds, color=colors, edgecolor="white",
           linewidth=0.5, capsize=3, error_kw={"elinewidth": 1})
    ax.set_ylabel(f"Cosine Similarity (destylized vs {label})")
    ax.set_title(f"SP: Destylized vs {label} per Operator")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    _save(fig, os.path.join(vis_dir, fname))


def plot_sp_destylized_sorted(
    sp_per_operator: Dict[str, Dict],
    vis_dir: str,
    *,
    vs: str = "root",
) -> None:
    sorted_items = sorted(sp_per_operator.items(), key=lambda x: x[1]["mean"], reverse=True)
    values = [v["mean"] for _, v in sorted_items]
    label  = "Root" if vs == "root" else "Mutated"
    fname  = f"sp_destylized_vs_{vs}_sorted.png"
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(values))
    ax.bar(x, values, color="#27ae60", edgecolor="none", width=1.0)
    ax.set_xlabel(f"Operator (sorted by SP destylized vs {label})")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"SP: Destylized vs {label} Distribution (sorted)")
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, os.path.join(vis_dir, fname))


# ---------------------------------------------------------------------------
# Cumulative recovery plot
# ---------------------------------------------------------------------------

def plot_cumulative_recovery(
    cumulative: Dict[int, Dict],
    vis_dir: str,
) -> None:
    depths   = sorted(cumulative.keys())
    rr       = [cumulative[d]["recovery_rate"] for d in depths]
    n_att    = [cumulative[d]["n_attempted"]   for d in depths]
    fig, ax1 = plt.subplots(figsize=FIGSIZE_LINE)
    ax2 = ax1.twinx()
    ax1.plot([str(d) for d in depths], rr, marker="o", color="#e74c3c",
             linewidth=2, label="Cumulative Recovery Rate")
    ax1.fill_between([str(d) for d in depths], rr, alpha=0.15, color="#e74c3c")
    ax2.bar([str(d) for d in depths], n_att, alpha=0.2, color="#3498db", label="N attempted")
    ax1.set_xlabel("Refinement Depth (cumulative up to d)")
    ax1.set_ylabel("Recovery Rate", color="#e74c3c")
    ax2.set_ylabel("N prompts attempted", color="#3498db")
    ax1.set_ylim(0, 1)
    ax1.set_title("Cumulative Recovery Rate vs Refinement Depth")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    _save(fig, os.path.join(vis_dir, "cumulative_recovery_by_depth.png"))


# ---------------------------------------------------------------------------
# Recovery vs SP scatter
# ---------------------------------------------------------------------------

def plot_recovery_vs_sp(
    edges: List[DestylizedEdge],
    embedder: "EmbeddingCache",
    vis_dir: str,
) -> None:
    """
    Scatter: x = SP(destylized, root), y = DES, colored by recovery.
    Shows whether prompts with high semantic similarity to root after
    destylization also have high DES (were more harmful surfaced).
    """
    sp_vals  = []
    des_vals = []
    colors   = []
    for e in edges:
        if e.destylized_prompt is None or e.destylized_fitness is None:
            continue
        sp_vals.append(embedder.cosine(e.root_prompt, e.destylized_prompt))
        des_vals.append(e.destylized_fitness - e.child_fitness)
        colors.append("#e74c3c" if e.recovered else "#3498db")

    if not sp_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sp_vals, des_vals, c=colors, alpha=0.4, s=10, linewidths=0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SP(destylized, root)")
    ax.set_ylabel("DES (destylized_fitness − child_fitness)")
    ax.set_title("Recovery vs Semantic Similarity to Root")
    from matplotlib.patches import Patch
    legend = [Patch(color="#e74c3c", label="Recovered"), Patch(color="#3498db", label="Not recovered")]
    ax.legend(handles=legend, fontsize=8)
    _save(fig, os.path.join(vis_dir, "recovery_vs_sp_scatter.png"))