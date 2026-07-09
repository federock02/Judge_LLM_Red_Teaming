"""
analyze_human_eval.py — Analyze the human-graded evaluation set (semantic preservation only).

Mirrors analyze_llm_evals.py, but the human blind sets only carry the
`semantic_preservation` field (1–5); `harm_severity` / `category_consistency` are left
null, so this script is restricted to semantic preservation.

For each judge the human grades live in:
    <base_dir>/<approach>_<judge>/
        *_blind_evaluation_set_300.json   # human grades: eval_id, semantic_preservation
        ground_truth_key_300.json         # automated key: eval_id, automated_semantic_sim, ...

The two files share the same (random-UUID) eval_ids because they were generated together,
so we join on `eval_id`. Note this is why the human set cannot be joined to the larger
llm_evaluation_results_*.json — those were prepared in a separate run with fresh UUIDs.

It reports, per judge and pooled:
  - Standalone human stats: N graded, mean/median/std of raw semantic_preservation, 1–5 histogram.
  - Human vs. automated: correlation, mean delta and MAE between the normalized human score
    and automated_semantic_sim.
  - Evolution by refinement_depth.
"""
import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_json(folder_path, must_contain):
    """Return the newest *.json in folder whose filename contains `must_contain`."""
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    hits = [f for f in json_files if must_contain.lower() in os.path.basename(f).lower()]
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def load_judge_frame(folder):
    """Load + join the human blind file and the automated key for one judge folder.

    Returns a DataFrame of graded rows (semantic_preservation not null), or None.
    """
    blind_path = find_json(folder, "blind")
    key_path = find_json(folder, "key")
    if blind_path is None:
        print(f"  [WARN] No blind file found in {folder}")
        return None
    if key_path is None:
        print(f"  [WARN] No ground-truth key found in {folder}")
        return None

    with open(blind_path, "r") as f:
        blind = pd.DataFrame(json.load(f))[["eval_id", "semantic_preservation"]]
    with open(key_path, "r") as f:
        key = pd.DataFrame(json.load(f))

    keep = [c for c in ["eval_id", "automated_semantic_sim", "refinement_depth",
                        "operator_name", "automated_judge_score"] if c in key.columns]
    key = key[keep]

    df = blind.merge(key, on="eval_id", how="inner")

    # Only rows the human actually graded.
    df["semantic_preservation"] = pd.to_numeric(df["semantic_preservation"], errors="coerce")
    n_total = len(df)
    df = df[df["semantic_preservation"].notna()].copy()
    print(f"  Loaded {blind_path} (+ key) -> {len(df)}/{n_total} graded rows")
    if df.empty:
        return None

    # 1–5 similarity -> [-1, 1], matching analyze_llm_evals.py's convention.
    df["norm_human_sim"] = (df["semantic_preservation"] - 3) / 2.0
    if "automated_semantic_sim" in df.columns:
        df["delta_sim"] = df["automated_semantic_sim"] - df["norm_human_sim"]
    return df


def generate_plots(df_judge, judge_name, output_dir):
    """Distribution, depth-evolution, and human-vs-automated scatter for one judge."""
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: raw semantic_preservation rating distribution ---
    plt.figure(figsize=(7, 5))
    sns.countplot(x="semantic_preservation", data=df_judge,
                  order=[1, 2, 3, 4, 5], color="#3498db")
    plt.title(f"{judge_name}: Human Semantic Preservation Ratings")
    plt.xlabel("semantic_preservation (1 = drift, 5 = exact match)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{judge_name}_human_sem_distribution.png"), dpi=300)
    plt.close()

    # --- PLOT 2: evolution by refinement depth (human vs automated, normalized) ---
    if {"refinement_depth", "automated_semantic_sim"}.issubset(df_judge.columns):
        plt.figure(figsize=(8, 6))
        depth_agg = df_judge.groupby("refinement_depth").agg(
            norm_human_sim=("norm_human_sim", "mean"),
            automated_semantic_sim=("automated_semantic_sim", "mean"),
        ).reset_index()

        sns.lineplot(data=depth_agg, x="refinement_depth", y="norm_human_sim",
                     marker="o", label="Human (normalized)", linewidth=2.5)
        sns.lineplot(data=depth_agg, x="refinement_depth", y="automated_semantic_sim",
                     marker="s", label="Automated Cosine Sim", linewidth=2.5)
        plt.title(f"{judge_name}: Semantic Preservation by Depth")
        plt.xlabel("Refinement Depth")
        plt.ylabel("Semantic Similarity Score")
        plt.xticks(depth_agg["refinement_depth"].astype(int))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{judge_name}_human_depth_evolution.png"), dpi=300)
        plt.close()

    # --- PLOT 3: human vs automated scatter with y=x reference ---
    if "automated_semantic_sim" in df_judge.columns:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x="norm_human_sim", y="automated_semantic_sim",
                        data=df_judge, alpha=0.5)
        lims = [-1.05, 1.05]
        plt.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.title(f"{judge_name}: Human vs Automated Semantic Sim")
        plt.xlabel("Human (normalized)")
        plt.ylabel("Automated Cosine Sim")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{judge_name}_human_vs_automated.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation (semantic preservation)")
    parser.add_argument("--base_dir", type=str, default="human_eval", help="Base directory of human-eval data")
    parser.add_argument("--approach", type=str, default="persona", help="Approach used (e.g., persona)")
    parser.add_argument("--judges", nargs="+", default=["LlamaGuard", "WildGuard", "ShieldGemma"],
                        help="List of judges")
    args = parser.parse_args()

    all_data = []
    report_lines = []

    def log(text=""):
        print(text)
        report_lines.append(text)

    print("=== LOADING HUMAN EVALUATION DATA ===")
    for judge in args.judges:
        folder = os.path.join(args.base_dir, f"{args.approach}_{judge}")
        if not os.path.isdir(folder):
            print(f"  [WARN] Folder not found: {folder}")
            continue
        df_judge = load_judge_frame(folder)
        if df_judge is None or df_judge.empty:
            print(f"  [WARN] {judge}: 0 graded rows, skipping.")
            continue
        df_judge["target_judge"] = judge
        df_judge["output_dir"] = folder
        print(f"  -> Generating plots for {judge} in {folder}...")
        generate_plots(df_judge, judge, folder)
        all_data.append(df_judge)

    if not all_data:
        print("[ERROR] No graded human data could be loaded. "
              "Fill in `semantic_preservation` in the blind files first. Exiting.")
        return

    df = pd.concat(all_data, ignore_index=True)
    has_auto = "automated_semantic_sim" in df.columns

    log(f"\n=== OVERALL HUMAN-EVAL STATISTICS ({args.approach.upper()}) ===")
    log(f"Total graded samples: {len(df)}")
    log(f"Graded per judge:\n{df['target_judge'].value_counts().to_string()}")
    log("-" * 40)

    log("\n=== STANDALONE HUMAN SEMANTIC PRESERVATION (raw 1-5) ===")
    for judge in args.judges:
        df_j = df[df["target_judge"] == judge]
        if len(df_j) == 0:
            continue
        sp = df_j["semantic_preservation"]
        log(f"{judge.ljust(15)}: n={len(df_j):<4} mean={sp.mean():.3f} "
            f"median={sp.median():.3f} std={sp.std():.3f}")
    log("\nRating histogram (count of 1..5 per judge):")
    hist = (df.groupby("target_judge")["semantic_preservation"]
              .value_counts().unstack(fill_value=0)
              .reindex(columns=[1, 2, 3, 4, 5], fill_value=0))
    log(hist.to_string())

    if has_auto:
        log("\n=== HUMAN vs AUTOMATED SEMANTIC SIM ===")
        log("(norm_human_sim = (semantic_preservation-3)/2 ; delta = automated - human)")
        for judge in args.judges:
            df_j = df[df["target_judge"] == judge]
            if len(df_j) == 0:
                continue
            corr = df_j["norm_human_sim"].corr(df_j["automated_semantic_sim"])
            delta = df_j["delta_sim"]
            log(f"{judge.ljust(15)}: corr={corr:.3f}  "
                f"mean_delta={delta.mean():+.3f}  MAE={delta.abs().mean():.3f}")
        corr_all = df["norm_human_sim"].corr(df["automated_semantic_sim"])
        log(f"{'POOLED'.ljust(15)}: corr={corr_all:.3f}  "
            f"mean_delta={df['delta_sim'].mean():+.3f}  MAE={df['delta_sim'].abs().mean():.3f}")

    if "refinement_depth" in df.columns:
        log("\n=== EVOLUTION BY REFINEMENT DEPTH (All Judges) ===")
        agg = {"semantic_preservation": "mean", "norm_human_sim": "mean"}
        if has_auto:
            agg["automated_semantic_sim"] = "mean"
        depth_stats = df.groupby("refinement_depth").agg(agg)
        log(depth_stats.round(3).to_string())

    report_filename = f"{args.approach}_human_eval_report.txt"
    report_path = os.path.join(args.base_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n[INFO] Plots successfully generated.")
    print(f"[SUCCESS] Human-eval report saved to: {report_path}")


if __name__ == "__main__":
    main()
