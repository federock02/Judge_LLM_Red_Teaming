"""
plots.py — shared plotting functions for transformation and persona analysis.
All plot functions save directly to vis_dir and close the figure.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from metrics import EmbeddingCache, MutationEdge, semantic_preservation


# ---------------------------------------------------------------------------
# Color map
# ---------------------------------------------------------------------------

def build_color_map(names: List[str], palette: str = "Paired") -> Dict[str, any]:
    colors = sns.color_palette(palette, n_colors=len(names) + 1)
    cmap = {name: colors[i] for i, name in enumerate(names)}
    cmap["Average"] = colors[-1]
    return cmap


# ---------------------------------------------------------------------------
# Internal bar helper
# ---------------------------------------------------------------------------

def _bar(
    names: List[str],
    values: List[float],
    colors: list,
    ylabel: str,
    title: str,
    filepath: str,
    *,
    yerr: Optional[List[float]] = None,
    yticks=None,
    add_average: bool = True,
    color_map: Optional[dict] = None,
):
    names  = list(names)
    values = list(values)
    colors = list(colors)

    if add_average:
        avg = float(np.mean(values))
        names.append("AVERAGE")
        values.append(avg)
        colors.append(color_map["Average"] if color_map else "black")
        if yerr is not None:
            yerr = list(yerr) + [float(np.std(values[:-1]))]

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, yerr=yerr, capsize=4 if yerr else 0, color=colors)

    if add_average:
        plt.axvline(x=len(names) - 1.5, color="black", linestyle="--", linewidth=2, alpha=0.6)

    plt.ylabel(ylabel, fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    if yticks is not None:
        plt.yticks(yticks, fontsize=18)
    else:
        plt.yticks(fontsize=18)
    plt.title(title, fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

def plot_asr(
    asr_per_operator: Dict[str, float],
    vis_dir: str,
    color_map: dict,
    *,
    level: str = "edge",
):
    names  = list(asr_per_operator.keys())
    values = list(asr_per_operator.values())
    colors = [color_map[n] for n in names]
    label  = "ASR_G" if level == "edge" else "ASR_R"
    title  = (
        "Global Attack Success Rate per Operator (ASR_G)"
        if level == "edge"
        else "Root-level Attack Success Rate per Operator (ASR_R)"
    )
    fname = f"{'edge' if level == 'edge' else 'root'}_level_asr_per_operator.png"
    _bar(names, values, colors, label, title, os.path.join(vis_dir, fname),
         add_average=True, color_map=color_map)


# ---------------------------------------------------------------------------
# Semantic Preservation
# ---------------------------------------------------------------------------

def plot_semantic_preservation(
    sp_per_operator: Dict[str, Dict[str, float]],
    vis_dir: str,
    color_map: dict,
    *,
    level: str = "edge",
):
    names  = list(sp_per_operator.keys())
    means  = [v["mean"] for v in sp_per_operator.values()]
    stds   = [v["std"]  for v in sp_per_operator.values()]
    colors = [color_map[n] for n in names]
    label  = "SP_G" if level == "edge" else "SP_R"
    title  = (
        "Global Semantic Preservation per Operator (SP_G)"
        if level == "edge"
        else "Root-level Semantic Preservation per Operator (SP_R)"
    )
    fname = f"{'edge' if level == 'edge' else 'root'}_level_sp_per_operator.png"
    _bar(names, means, colors, label, title, os.path.join(vis_dir, fname),
         yerr=stds, add_average=True, color_map=color_map)


def plot_semantic_preservation_violin(
    grouped: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
    vis_dir: str,
    color_map: dict,
    *,
    reference: str = "root",
):
    labels = list(grouped.keys())
    data   = [semantic_preservation(es, embedder, reference=reference) for es in grouped.values()]

    plt.figure(figsize=(24, 10))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_map[labels[i]])
        pc.set_alpha(0.7)

    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=22)
    plt.ylabel("SP_G", fontsize=24)
    plt.yticks(fontsize=20)
    plt.title("SP Distribution per Operator (SP_G)", fontsize=22)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "sp_violin_per_operator.png"), dpi=200)
    plt.close()


def plot_sp_sorted(
    sorted_data: List[tuple],
    vis_dir: str,
    *,
    level: str = "edge",
    color: str = "#3498db",
):
    """Sorted-bar distribution for large operator sets (e.g. 20k personas)."""
    means = [v for _, v in sorted_data]
    avg   = float(np.mean(means)) if means else 0.0
    label = "SP_G" if level == "edge" else "SP_R"
    fname = f"{'edge' if level == 'edge' else 'root'}_sp_sorted.png"

    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(len(means)), means, color=color, width=0.8, linewidth=0)
    plt.axhline(avg, color="red", linestyle="--", linewidth=2,
                label=f"AVERAGE {label} ({avg:.2f})")
    plt.ylabel(label, fontsize=22)
    plt.xlabel(f"Operators sorted by {label}", fontsize=22)
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.1)
    plt.title(f"Distribution of {label} across all Operators", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, fname), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# JCO
# ---------------------------------------------------------------------------

def plot_jco(
    jco_per_operator: Dict[str, float],
    vis_dir: str,
    color_map: dict,
    *,
    level: str = "edge",
):
    names  = list(jco_per_operator.keys())
    values = list(jco_per_operator.values())
    colors = [color_map[n] for n in names]
    label  = "JCO_G" if level == "edge" else "JCO_R"
    title  = (
        "Global JCO per Operator (JCO_G)"
        if level == "edge"
        else "Root-level JCO per Operator (JCO_R)"
    )
    fname = f"{'edge' if level == 'edge' else 'root'}_level_jco_per_operator.png"
    _bar(names, values, colors, label, title, os.path.join(vis_dir, fname),
         yticks=np.arange(0, 0.9, 0.1), add_average=True, color_map=color_map)


# ---------------------------------------------------------------------------
# MEI
# ---------------------------------------------------------------------------

def plot_mei(mei_per_operator: Dict[str, float], vis_dir: str, color_map: dict):
    names  = list(mei_per_operator.keys())
    values = list(mei_per_operator.values())
    colors = [color_map[n] for n in names]
    _bar(names, values, colors, "MEI", "Mutation Efficiency Index per Operator (MEI)",
         os.path.join(vis_dir, "mei_per_operator.png"),
         add_average=True, color_map=color_map)


# ---------------------------------------------------------------------------
# Depth of Success
# ---------------------------------------------------------------------------

def plot_depth_of_success(
    depth_stats: Dict[str, Dict[str, float]],
    vis_dir: str,
    color_map: dict,
    *,
    level: str = "edge",
):
    names  = [n for n, s in depth_stats.items() if s]
    means  = [depth_stats[n]["mean"] for n in names]
    colors = [color_map[n] for n in names]
    label  = "d_G" if level == "edge" else "d_R"
    title  = (
        "Global Depth of Success per Operator (d_G)"
        if level == "edge"
        else "Root-level Depth of Success per Operator (d_R)"
    )
    fname = f"{'edge' if level == 'edge' else 'root'}_level_depth_per_operator.png"
    _bar(names, means, colors, label, title, os.path.join(vis_dir, fname),
         yticks=np.arange(0, 6), add_average=True, color_map=color_map)


def plot_depth_sorted(sorted_data: List[tuple], vis_dir: str, *, color: str = "#079D68"):
    depths = [d for _, d in sorted_data]
    succ   = [d for d in depths if d > 0]
    avg    = float(np.mean(succ)) if succ else 0.0

    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(len(depths)), depths, color=color, width=1.0, linewidth=0, edgecolor="none")
    plt.axhline(avg, color="red", linestyle="--", linewidth=2,
                label=f"AVG Depth of Success ({avg:.2f})")
    plt.ylabel("Refinement Depth", fontsize=22)
    plt.xlabel("Operators sorted by depth", fontsize=22)
    plt.xticks([])
    plt.yticks(range(0, 6), fontsize=18)
    plt.ylim(0, 5.5)
    plt.title("Distribution of Edge-level Depth of Success", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_depth_sorted.png"), dpi=300)
    plt.close()


def plot_root_depth_sorted(
    root_data: List[tuple],
    vis_dir: str,
    *,
    label: str = "min",
    color: str = "#9b59b6",
):
    sorted_data = sorted(root_data, key=lambda x: x[1], reverse=True)
    depths = [d for _, d in sorted_data]
    avg    = float(np.mean(depths)) if depths else 0.0
    ylabel = "Minimum Depth to Success" if label == "min" else "Average Depth to Success"

    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(len(depths)), depths, color=color, width=0.8, linewidth=0)
    plt.axhline(avg, color="red", linestyle="--", linewidth=2, label=f"AVG ({avg:.2f})")
    plt.ylabel(ylabel, fontsize=22)
    plt.xlabel("Unique root prompts, sorted by depth", fontsize=22)
    plt.xticks([])
    plt.yticks(range(0, 6), fontsize=18)
    plt.ylim(0, 5.5)
    plt.title(f"Root-level {label.capitalize()} Depth of Success", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"root_level_{label}_depth_sorted.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Semantic Drift
# ---------------------------------------------------------------------------

def plot_drift_rate(
    drift_stats: Dict[str, Dict[str, float]],
    vis_dir: str,
    color_map: dict,
):
    names  = list(drift_stats.keys())
    means  = [v["mean"] for v in drift_stats.values()]
    stds   = [v["std"]  for v in drift_stats.values()]
    colors = [color_map[n] for n in names]
    _bar(names, means, colors, "ΔSP", "Average Semantic Drift per Operator (ΔSP)",
         os.path.join(vis_dir, "semantic_drift_per_operator.png"),
         yerr=stds, yticks=np.arange(0, 0.7, 0.1),
         add_average=True, color_map=color_map)


def plot_drift_sorted(sorted_data: List[tuple], vis_dir: str, *, color: str = "#e74c3c"):
    means = [v for _, v in sorted_data]
    avg   = float(np.mean(means)) if means else 0.0

    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(len(means)), means, color=color, width=1.0, linewidth=0, edgecolor="none")
    plt.axhline(avg, color="black", linestyle="--", linewidth=2,
                label=f"AVERAGE ΔSP ({avg:.3f})")
    plt.ylabel("Semantic Drift (ΔSP)", fontsize=22)
    plt.xlabel("Operators sorted by drift", fontsize=20)
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.title("Semantic Drift Rate (sorted)", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "drift_sorted.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# ASR vs SP
# ---------------------------------------------------------------------------

def plot_asr_sp_heatmap(
    asr_per_operator: Dict[str, float],
    sp_per_operator:  Dict[str, Dict[str, float]],
    vis_dir: str,
    *,
    level: str = "edge",
):
    rows = [
        [name, asr_per_operator[name], sp_per_operator[name]["mean"]]
        for name in asr_per_operator
    ]
    rows.append([
        "AVERAGE",
        float(np.mean(list(asr_per_operator.values()))),
        float(np.mean([v["mean"] for v in sp_per_operator.values()])),
    ])
    df = pd.DataFrame(rows, columns=["Operator", "ASR", "SP"]).set_index("Operator")

    sns.set_context("talk", font_scale=2.2)
    plt.figure(figsize=(16, max(6, 0.9 * len(df))), dpi=200)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu",
                     linewidths=0.5, annot_kws={"size": 26},
                     cbar_kws={"shrink": 0.8})
    label = "Global" if level == "edge" else "Root-level"
    plt.title(f"{label} ASR and SP per Operator", fontsize=32, pad=25)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26, rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{level}_asr_sp_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_asr_vs_sp(
    asr_per_operator: Dict[str, float],
    sp_per_operator:  Dict[str, Dict[str, float]],
    vis_dir: str,
    color_map: dict,
    *,
    level: str = "edge",
):
    plt.figure(figsize=(12, 10))
    for name in asr_per_operator:
        if name not in sp_per_operator:
            continue
        plt.scatter(
            sp_per_operator[name]["mean"],
            asr_per_operator[name],
            color=color_map.get(name, "gray"),
            label=name,
            s=300,
        )
    avg_sp  = float(np.mean([v["mean"] for v in sp_per_operator.values()]))
    avg_asr = float(np.mean(list(asr_per_operator.values())))
    plt.scatter(avg_sp, avg_asr, color=color_map["Average"], label="AVERAGE", s=450)
    plt.xlabel("Semantic Preservation (⟶ better)", fontsize=20)
    plt.ylabel("Attack Success Rate (⟶ better)", fontsize=20)
    plt.title("ASR vs Semantic Preservation", fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{level}_asr_vs_sp.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Success vs SP scatter
# ---------------------------------------------------------------------------

def plot_success_vs_sp(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    vis_dir: str,
):
    success_sp = [embedder.cosine(e.root_prompt, e.child_prompt) for e in edges if e.success]
    failure_sp = [embedder.cosine(e.root_prompt, e.child_prompt) for e in edges if not e.success]

    def jitter(val, n):
        return np.random.normal(val, 0.04, n)

    plt.figure(figsize=(12, 10))
    plt.scatter(jitter(0, len(failure_sp)), failure_sp,
                color="red",  alpha=0.1, s=10, label="Failure", rasterized=True)
    plt.scatter(jitter(1, len(success_sp)), success_sp,
                color="blue", alpha=0.1, s=10, label="Success", rasterized=True)

    avg_f = float(np.mean(failure_sp)) if failure_sp else 0.0
    avg_s = float(np.mean(success_sp)) if success_sp else 0.0
    for x, avg, c, lbl in [(0, avg_f, "red", "Failure"), (1, avg_s, "blue", "Success")]:
        plt.plot(x, avg, marker="D", markersize=15, color="white",
                 markeredgecolor="black", markeredgewidth=2)
        plt.plot(x, avg, marker="D", markersize=12, color=c,
                 label=f"Avg {lbl} ({avg:.2f})")

    plt.title("Semantic Preservation vs. Attack Success", fontsize=24)
    plt.ylabel("Global Semantic Preservation (SP_G)", fontsize=22)
    plt.xlabel("Attack Success", fontsize=22)
    plt.xticks([0, 1], ["Failure", "Success"], fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.1)
    plt.xlim(-0.5, 1.5)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(fontsize=16, loc="lower center", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "success_vs_sp_scatter.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Cumulative metrics vs depth
# ---------------------------------------------------------------------------

def plot_cumulative_metrics_vs_depth(
    depth_stats: Dict[int, Dict[str, float]],
    vis_dir: str,
):
    depths    = sorted(depth_stats.keys())
    asr_vals  = [depth_stats[d]["asr"]         for d in depths]
    pres_vals = [depth_stats[d]["preservation"] for d in depths]
    jco_vals  = [depth_stats[d]["jco"]          for d in depths]

    plt.figure(figsize=(12, 8))
    plt.plot(depths, asr_vals,  marker="o", linewidth=3, label="Cumulative ASR")
    plt.plot(depths, pres_vals, marker="s", linewidth=3, label="Cumulative SP")
    plt.plot(depths, jco_vals,  marker="^", linewidth=3, label="Cumulative JCO")
    plt.xlabel("Refinement Depth", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.title("Cumulative Attack Metrics vs Refinement Depth", fontsize=24)
    plt.xticks(depths, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "cumulative_metrics_vs_depth.png"), dpi=200)
    plt.close()