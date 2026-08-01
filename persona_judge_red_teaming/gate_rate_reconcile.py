"""Reconcile the edge-level semantic-gate survival rate with the rate measured on the
human/LLM evaluation sample.

The archive-wide metrics in metrics.py count *every* bypassing mutation edge.
sample_for_human_eval.py first deduplicates candidate edges by unique `child_prompt`
text (lines 42-48) and only then draws its 100-edge sample. If duplicate children are
not distributed uniformly over the similarity range, the two populations have different
gate survival rates -- which is the discrepancy this script measures.

Reuses load_edges / EmbeddingCache / _is_valid_success unchanged, so (a) below must
reproduce edge_level_asr_star / edge_level_bypass_rate from the main run exactly.
"""
import argparse, json, os, collections
import numpy as np

from analyze_results import load_edges
from metrics import EmbeddingCache, TAU_SP, _is_valid_success, group_edges_by_operator


def survival(edges, embedder, tau_sp):
    """(gate survival among bypasses, n_bypasses, asr_star_edge, bypass_rate_edge)."""
    if not edges:
        return float("nan"), 0, 0.0, 0.0
    byp = [e for e in edges if e.success]
    valid = [e for e in byp if _is_valid_success(e, embedder, tau_sp)]
    rate = len(valid) / len(byp) if byp else float("nan")
    return rate, len(byp), len(valid) / len(edges), len(byp) / len(edges)


def dedup_by_child(edges):
    """Mirror sample_for_human_eval.py:42-48 -- keep first edge per unique child text."""
    seen, out = set(), []
    for e in edges:
        if e.child_prompt not in seen:
            seen.add(e.child_prompt)
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_folders", nargs="+", required=True)
    ap.add_argument("--tau_sp", type=float, default=TAU_SP)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [os.path.join(f, "red_teaming_archive.json") for f in args.results_folders]
    paths = [p for p in paths if os.path.isfile(p)]
    edges = load_edges(paths)

    embedder = EmbeddingCache(device=args.device)
    embedder.batch_embed(list({t for e in edges
                               for t in (e.root_prompt, e.parent_prompt, e.child_prompt)}))

    res = {"tau_sp": args.tau_sp, "n_edges": len(edges)}

    # (a) every bypassing edge -- must match the published edge-level numbers
    r, n, asr, byp = survival(edges, embedder, args.tau_sp)
    res["all_edges"] = {"survival": r, "n_bypasses": n,
                        "edge_level_asr_star": asr, "edge_level_bypass_rate": byp}

    # (b) deduplicated by child text, as the eval sample was drawn
    ded = dedup_by_child(edges)
    r_d, n_d, _, _ = survival(ded, embedder, args.tau_sp)
    res["dedup_by_child"] = {"survival": r_d, "n_bypasses": n_d,
                             "n_edges": len(ded)}

    # duplication factor among bypasses only
    byp_all = [e for e in edges if e.success]
    byp_ded = [e for e in ded if e.success]
    res["bypass_duplication_factor"] = (len(byp_all) / len(byp_ded)) if byp_ded else float("nan")

    # (c) per operator: survival, share of bypasses, duplication
    per_op = {}
    for name, es in group_edges_by_operator(edges).items():
        r_o, n_o, asr_o, byp_o = survival(es, embedder, args.tau_sp)
        es_d = dedup_by_child(es)
        r_od, n_od, _, _ = survival(es_d, embedder, args.tau_sp)
        per_op[name] = {
            "survival_all": r_o, "n_bypasses": n_o,
            "survival_dedup": r_od, "n_bypasses_dedup": n_od,
            "dup_factor": (n_o / n_od) if n_od else float("nan"),
            "edge_level_asr_star": asr_o, "edge_level_bypass_rate": byp_o,
            "mean_sp_of_bypasses": float(np.mean(
                [embedder.cosine(e.root_prompt, e.child_prompt) for e in es if e.success]
            )) if n_o else float("nan"),
        }
    res["per_operator"] = per_op

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n===== GATE-RATE RECONCILIATION =====")
    print(f"tau_SP = {args.tau_sp}   edges = {len(edges)}")
    print(f"(a) all bypassing edges     : survival = {res['all_edges']['survival']:.4f}"
          f"  (n={n})")
    print(f"    -> edge_level_asr_star  = {asr:.4f}   bypass_rate = {byp:.4f}"
          f"   [must match published run]")
    print(f"(b) dedup by child_prompt   : survival = {r_d:.4f}  (n={n_d})")
    print(f"    duplication factor among bypasses = {res['bypass_duplication_factor']:.3f}")
    print(f"    delta (a)-(b) = {res['all_edges']['survival'] - r_d:+.4f}")
    print("\n(c) per operator (sorted by dup factor):")
    print(f"{'operator':<34}{'surv_all':>9}{'surv_ded':>9}{'dup':>7}{'n_byp':>9}{'meanSP':>8}")
    for name, d in sorted(per_op.items(), key=lambda kv: -(kv[1]['dup_factor'] or 0)):
        print(f"{name[:33]:<34}{d['survival_all']:>9.4f}{d['survival_dedup']:>9.4f}"
              f"{d['dup_factor']:>7.2f}{d['n_bypasses']:>9d}{d['mean_sp_of_bypasses']:>8.4f}")
    print(f"\n[WROTE] {args.out}")


if __name__ == "__main__":
    main()
