"""
analyze_ablation.py — Analysis script for comparing red-teaming ablation configs.

The ablation is a 3x2 matrix over two independent axes:
  - score_mode  : "normal" / "random" / "none"  (what judge score the attacker sees)
  - use_history : history / no_history           (whether the previous attempt is shown)

giving six archive configs:
  normal_history, normal_no_history,
  random_history, random_no_history,
  none_history,   none_no_history

For each config it computes:
  - ASR@1 (Attack Success Rate at depth 1)
  - ASR@<5 (Attack Success Rate at depth < 5)
  - Average / Median Depth of Success
  - Overall ASR

On top of the per-config table it also prints (and saves to a text report) two
marginal summaries that isolate each axis:
  - History effect  : Overall / ASR@1 with vs. without history, per score mode.
  - Score-mode effect: Overall ASR per score mode, deltas vs. the `normal` baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ablation configuration matrix (score_mode x use_history)
# ---------------------------------------------------------------------------
SCORE_MODES = ["normal", "random", "none"]
HISTORY_KEYS = ["history", "no_history"]

# Ordered so the two history variants of each score mode sit next to each other.
CONFIGS = [f"{sm}_{hk}" for sm in SCORE_MODES for hk in HISTORY_KEYS]

# One color family per score mode; no_history is drawn lighter / hatched.
SCORE_COLORS = {"normal": "#2ecc71", "random": "#f1c40f", "none": "#e74c3c"}


def config_score_mode(config: str) -> str:
    """'normal_no_history' -> 'normal'."""
    return config.split("_", 1)[0]


def config_uses_history(config: str) -> bool:
    """'normal_no_history' -> False, 'normal_history' -> True."""
    return config.split("_", 1)[1] == "history"


def pretty_label(config: str, short: bool = False) -> str:
    sm = config_score_mode(config).capitalize()
    if short:
        return f"{sm}\n{'+H' if config_uses_history(config) else '−H'}"
    return f"{sm} ({'with history' if config_uses_history(config) else 'no history'})"


# ---------------------------------------------------------------------------
# Data Loading and Parsing
# ---------------------------------------------------------------------------

def load_ablation_data(filepaths) -> dict:
    """Loads one or more JSON archives and calculates core metrics per ablation config.

    Multiple archives are POOLED into a single analysis: every root entry across all
    files is accumulated into the same trackers, so metrics reflect the combined sample
    (useful when several experiment runs are meant to be treated as one larger dataset).
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    # Initialize metric trackers for each config
    raw_stats = {
        config: {
            "total_prompts": 0,
            "success_at_1": 0,
            "success_under_5": 0,
            "overall_success": 0,
            "success_depths": []
        } for config in CONFIGS
    }

    for filepath in filepaths:
        print(f"[LOAD] Parsing data from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)

        for key, val in data.items():
            # Skip metadata
            if key.startswith("_"):
                continue

            for config in CONFIGS:
                if config not in val:
                    continue

                raw_stats[config]["total_prompts"] += 1
                attempts = val[config].get("attempts", [])

                # Track the minimum depth at which a success occurred for this root
                min_success_depth = float('inf')

                for attempt in attempts:
                    ref_iter = attempt.get("refinement_iter", 1)
                    success_flags = attempt.get("success", [])

                    # If any of the mutated prompts at this depth succeeded
                    if any(success_flags):
                        if ref_iter < min_success_depth:
                            min_success_depth = ref_iter

                # If a success was found, update our aggregations
                if min_success_depth != float('inf'):
                    raw_stats[config]["overall_success"] += 1
                    raw_stats[config]["success_depths"].append(min_success_depth)

                    if min_success_depth == 1:
                        raw_stats[config]["success_at_1"] += 1
                    if min_success_depth < 5:
                        raw_stats[config]["success_under_5"] += 1

    return calculate_final_metrics(raw_stats)


def calculate_final_metrics(raw_stats: dict) -> dict:
    """Converts raw counts into percentages and averages."""
    metrics = {}
    for config in CONFIGS:
        stats = raw_stats[config]
        total = stats["total_prompts"]

        if total == 0:
            metrics[config] = None
            continue

        depths = stats["success_depths"]

        metrics[config] = {
            "Total Runs": total,
            "ASR@1 (%)": (stats["success_at_1"] / total) * 100,
            "ASR@<5 (%)": (stats["success_under_5"] / total) * 100,
            "Overall ASR (%)": (stats["overall_success"] / total) * 100,
            "Avg Depth of Success": float(np.mean(depths)) if depths else 0.0,
            "Median Depth of Success": float(np.median(depths)) if depths else 0.0,
            "_raw_depths": depths  # Kept for boxplots
        }

    return metrics


# ---------------------------------------------------------------------------
# Marginal summaries (isolate one ablation axis at a time)
# ---------------------------------------------------------------------------

def _metric_or_none(metrics: dict, config: str, field: str):
    data = metrics.get(config)
    return data[field] if data is not None else None


def _fmt(val) -> str:
    return f"{val:6.2f}" if val is not None else "   n/a"


def _fmt_delta(a, b) -> str:
    if a is None or b is None:
        return "   n/a"
    return f"{a - b:+6.2f}"


def summarize_history_effect(metrics: dict, log) -> None:
    """Overall / ASR@1 with vs. without history, per score mode (the history axis)."""
    present_modes = [sm for sm in SCORE_MODES
                     if metrics.get(f"{sm}_history") is not None
                     or metrics.get(f"{sm}_no_history") is not None]
    if not present_modes:
        return

    log("\n=== HISTORY EFFECT (per score mode) ===")
    log("Overall ASR: how much the previous-attempt context helps, holding score mode fixed.")
    log(f"{'Score mode':<12}{'+History':>10}{'−History':>10}{'Δ(+H − −H)':>12}")
    hist_vals, nohist_vals = [], []
    for sm in present_modes:
        h = _metric_or_none(metrics, f"{sm}_history", "Overall ASR (%)")
        n = _metric_or_none(metrics, f"{sm}_no_history", "Overall ASR (%)")
        if h is not None:
            hist_vals.append(h)
        if n is not None:
            nohist_vals.append(n)
        log(f"{sm.capitalize():<12}{_fmt(h):>10}{_fmt(n):>10}{_fmt_delta(h, n):>12}")

    avg_h = float(np.mean(hist_vals)) if hist_vals else None
    avg_n = float(np.mean(nohist_vals)) if nohist_vals else None
    log(f"{'AVERAGE':<12}{_fmt(avg_h):>10}{_fmt(avg_n):>10}{_fmt_delta(avg_h, avg_n):>12}")

    log("\nASR@1 (same axis, depth-1 successes only):")
    log(f"{'Score mode':<12}{'+History':>10}{'−History':>10}{'Δ(+H − −H)':>12}")
    for sm in present_modes:
        h = _metric_or_none(metrics, f"{sm}_history", "ASR@1 (%)")
        n = _metric_or_none(metrics, f"{sm}_no_history", "ASR@1 (%)")
        log(f"{sm.capitalize():<12}{_fmt(h):>10}{_fmt(n):>10}{_fmt_delta(h, n):>12}")


def summarize_score_mode_effect(metrics: dict, log) -> None:
    """Overall ASR per score mode within each history setting, vs. the `normal` baseline.

    The `normal` column is the true-score baseline; `random`/`none` isolate how much
    the attacker relies on an informative judge score.
    """
    log("\n=== SCORE-MODE EFFECT (per history setting) ===")
    log("Overall ASR per score mode; Δ columns are relative to the `normal` (true-score) baseline.")
    for hk in HISTORY_KEYS:
        present = [sm for sm in SCORE_MODES if metrics.get(f"{sm}_{hk}") is not None]
        if not present:
            continue
        baseline = _metric_or_none(metrics, f"normal_{hk}", "Overall ASR (%)")
        label = "with history" if hk == "history" else "no history"
        log(f"\n  [{label}]  baseline = normal")
        log(f"  {'Score mode':<12}{'Overall ASR':>12}{'Δ vs normal':>14}")
        for sm in present:
            v = _metric_or_none(metrics, f"{sm}_{hk}", "Overall ASR (%)")
            delta = "  (baseline)" if sm == "normal" else _fmt_delta(v, baseline)
            log(f"  {sm.capitalize():<12}{_fmt(v):>12}{delta:>14}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_metrics(metrics: dict, vis_dir: str):
    """Generates and saves bar charts and box plots comparing the six configs."""
    os.makedirs(vis_dir, exist_ok=True)

    # Keep canonical order, dropping configs that had no data
    valid = [c for c in CONFIGS if metrics.get(c) is not None]
    if not valid:
        print("[ERROR] No valid config data found to plot.")
        return

    # 1. Bar Chart: ASR metrics per config (grouped by score mode / history)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(valid))
    width = 0.25

    asr_1 = [metrics[c]["ASR@1 (%)"] for c in valid]
    asr_5 = [metrics[c]["ASR@<5 (%)"] for c in valid]
    asr_all = [metrics[c]["Overall ASR (%)"] for c in valid]

    ax.bar(x - width, asr_1, width, label='ASR@1', color='#3498db')
    ax.bar(x, asr_5, width, label='ASR@<5', color='#9b59b6')
    ax.bar(x + width, asr_all, width, label='Overall ASR', color='#34495e')

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Attack Success Rates by Ablation Config (score mode x history)')
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(c, short=True) for c in valid])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(os.path.join(vis_dir, "asr_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Grouped Bar Chart: Overall ASR, history vs no-history per score mode
    fig, ax = plt.subplots(figsize=(9, 6))
    present_modes = [sm for sm in SCORE_MODES
                     if any(config_score_mode(c) == sm for c in valid)]
    xm = np.arange(len(present_modes))
    bar_w = 0.38

    def overall_for(sm, use_hist):
        cfg = f"{sm}_{'history' if use_hist else 'no_history'}"
        return metrics[cfg]["Overall ASR (%)"] if metrics.get(cfg) is not None else 0.0

    hist_vals = [overall_for(sm, True) for sm in present_modes]
    nohist_vals = [overall_for(sm, False) for sm in present_modes]

    ax.bar(xm - bar_w / 2, hist_vals, bar_w, label='With history',
           color=[SCORE_COLORS[sm] for sm in present_modes])
    ax.bar(xm + bar_w / 2, nohist_vals, bar_w, label='No history',
           color=[SCORE_COLORS[sm] for sm in present_modes], alpha=0.55, hatch='//')

    ax.set_ylabel('Overall ASR (%)')
    ax.set_title('Overall ASR: effect of history conditioning per score mode')
    ax.set_xticks(xm)
    ax.set_xticklabels([sm.capitalize() for sm in present_modes])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(os.path.join(vis_dir, "asr_history_effect.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Box Plot: Depth of Success Distribution across the six configs
    fig, ax = plt.subplots(figsize=(12, 6))
    depth_data = [metrics[c]["_raw_depths"] for c in valid]

    bplot = ax.boxplot(depth_data, patch_artist=True)
    # Set labels after the call for matplotlib version robustness (tick_labels/labels
    # kwargs differ across versions); boxplot positions default to 1..N.
    ax.set_xticks(np.arange(1, len(valid) + 1))
    ax.set_xticklabels([pretty_label(c, short=True) for c in valid])

    for patch, config in zip(bplot['boxes'], valid):
        patch.set_facecolor(SCORE_COLORS.get(config_score_mode(config), "#bdc3c7"))
        if not config_uses_history(config):
            patch.set_alpha(0.55)
            patch.set_hatch('//')

    ax.set_ylabel('Refinement Iteration (Depth)')
    ax.set_title('Distribution of Success Depths by Ablation Config')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(os.path.join(vis_dir, "depth_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Visualizations saved to {vis_dir}/")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation results across the 3x2 score-mode x history matrix"
    )
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="Path(s) to the JSON results file(s). Multiple files are pooled "
                             "into a single combined analysis.")
    parser.add_argument("--vis_dir", type=str, default="ablation_plots", help="Directory to save plots")
    parser.add_argument("--approach", type=str, default="persona",
                        help="Approach name, used only to title the saved report")
    parser.add_argument("--report_path", type=str, default=None,
                        help="Where to save the text report (default: <vis_dir>/ablation_report.txt)")

    args = parser.parse_args()

    metrics = load_ablation_data(args.input)

    # Capture everything printed below into a saved report as well.
    report_lines = []

    def log(text=""):
        print(text)
        report_lines.append(text)

    log(f"=== ABLATION ANALYSIS ({args.approach.upper()}) ===")
    log(f"Input archive(s) [{len(args.input)}, pooled]:")
    for p in args.input:
        log(f"  - {p}")

    log("\n--- Per-config results (score mode x history) ---")
    for config in CONFIGS:
        data = metrics.get(config)
        if data is None:
            continue
        log(f"\nConfig: {pretty_label(config)}")
        log(f"  Total Runs:             {data['Total Runs']}")
        log(f"  ASR@1:                  {data['ASR@1 (%)']:.2f}%")
        log(f"  ASR@<5:                 {data['ASR@<5 (%)']:.2f}%")
        log(f"  Overall ASR:            {data['Overall ASR (%)']:.2f}%")
        log(f"  Avg Depth of Success:   {data['Avg Depth of Success']:.2f}")
        log(f"  Median Depth of Success:{data['Median Depth of Success']:.2f}")

    # Marginal summaries that isolate each ablation axis.
    summarize_history_effect(metrics, log)
    summarize_score_mode_effect(metrics, log)

    plot_metrics(metrics, args.vis_dir)

    # Persist the report next to the plots by default.
    os.makedirs(args.vis_dir, exist_ok=True)
    report_path = args.report_path or os.path.join(args.vis_dir, "ablation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[SUCCESS] Ablation report saved to: {report_path}")


if __name__ == "__main__":
    main()
