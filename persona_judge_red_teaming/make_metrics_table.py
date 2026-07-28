"""
make_metrics_table.py — Combine per-judge metrics JSONs into a single table.

Reads <vis_dir>/<judge>/persona_attack_metrics.json (or
transformation_attack_metrics.json) for each judge subdirectory and
produces both a printed table and a CSV.

Usage:
    python make_metrics_table.py --vis_dir visualizations/
    python make_metrics_table.py --vis_dir visualizations/ --out metrics_table.csv
    python make_metrics_table.py --vis_dir visualizations/ --approach transformation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(data: dict) -> Dict[str, Optional[float]]:
    """
    Flatten the global metrics dict into a single-level dict of
    metric_name -> float, ignoring per-operator breakdowns.
    """
    g = data.get("global", data)  # support both wrapped and flat formats

    def get(key, subkey=None):
        val = g.get(key)
        if val is None:
            return None
        if subkey is not None:
            return val.get(subkey)
        return val

    return {
        "N edges":          get("num_edges"),
        "N roots":          get("num_unique_root_prompts"),
        "tau_SP":           get("tau_sp"),
        # ASR*: evasion AND semantic preservation. The headline attack success rate.
        "ASR* (edge)":      get("edge_level_asr_star"),
        "ASR* (root)":      get("root_level_asr_star"),
        "ASR*@1":           get("asr_star_at_1"),
        "ASR*@5":           get("asr_star_at_5"),
        # Bypass rate: evasion only. Reported as an upper bound on ASR*.
        "Bypass (edge)":    get("edge_level_bypass_rate"),
        "Bypass (root)":    get("root_level_bypass_rate"),
        "Bypass@1":         get("bypass_rate_at_1"),
        "Bypass@5":         get("bypass_rate_at_5"),
        "SP (edge) mean":   get("edge_level_semantic_preservation", "mean"),
        "SP (edge) std":    get("edge_level_semantic_preservation", "std"),
        "SP (root) mean":   get("root_level_semantic_preservation", "mean"),
        "SP (root) std":    get("root_level_semantic_preservation", "std"),
        "JCO (edge)":       get("edge_level_jco"),
        "JCO (root)":       get("root_level_jco"),
        "Depth (edge) mean":get("edge_level_depth_of_success",  "mean"),
        "Depth (edge) max": get("edge_level_depth_of_success",  "max"),
        "Depth (root) mean":get("root_level_depth_of_success",  "mean"),
        "Depth (root) max": get("root_level_depth_of_success",  "max"),
        "MEI":              get("mutation_efficiency_index"),
        "Drift mean":       get("semantic_drift_rate", "mean"),
        "Drift std":        get("semantic_drift_rate", "std"),
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_val(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.4f}"


def print_table(rows: List[Dict], judges: List[str], metrics: List[str]) -> None:
    col_w = max(len(m) for m in metrics) + 2
    judge_w = max(max(len(j) for j in judges), 12) + 2

    header = f"{'Metric':<{col_w}}" + "".join(f"{j:>{judge_w}}" for j in judges)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in metrics:
        row = f"{metric:<{col_w}}"
        for judge in judges:
            val = rows[judge].get(metric)
            row += f"{format_val(val):>{judge_w}}"
        print(row)

    print("=" * len(header) + "\n")


def write_csv(rows: Dict, judges: List[str], metrics: List[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + judges)
        for metric in metrics:
            writer.writerow(
                [metric] + [format_val(rows[judge].get(metric)) for judge in judges]
            )
    print(f"[TABLE] CSV saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-judge metrics JSONs into a comparison table."
    )
    parser.add_argument(
        "--vis_dir", type=str, required=True,
        help="Visualizations root directory containing one subfolder per judge.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional path for CSV output. Default: <vis_dir>/metrics_table.csv",
    )
    args = parser.parse_args()

    print(f"Creating table from data in {args.vis_dir}...")

    vis_dir  = Path(args.vis_dir)
    filename = "persona_attack_metrics.json"
    out_path = args.out or str(vis_dir / "metrics_table.csv")

    # Discover judge subdirectories
    judges = sorted([
        d.name for d in vis_dir.iterdir()
        if d.is_dir() and (d / filename).exists()
    ])

    if not judges:
        print(f"[TABLE] No '{filename}' found in any subdirectory of {vis_dir}")
        return

    print(f"[TABLE] Found judges: {judges}")

    rows: Dict[str, Dict] = {}
    for judge in judges:
        path = vis_dir / judge / filename
        with open(path) as f:
            data = json.load(f)
        rows[judge] = extract_metrics(data)

    # Use metric order from first judge, keeping all keys
    metrics = list(next(iter(rows.values())).keys())

    print_table(rows, judges, metrics)
    write_csv(rows, judges, metrics, out_path)


if __name__ == "__main__":
    main()