"""
persona_clustering.py — HDBSCAN-based persona clustering and UMAP visualization.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

from metrics import EmbeddingCache, MutationEdge

# ---------------------------------------------------------------------------
# Field weights for weighted persona embedding
# ---------------------------------------------------------------------------

FIELD_WEIGHTS: Dict[str, float] = {
    "persona_obfuscation_strategy": 3.0,
    "persona_domain":               2.5,
    "persona_style":                1.0,
    "persona_tone":                 0.5,
    "persona_name":                 0.2,
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def persona_to_weighted_embedding(
    meta: dict,
    embedder: EmbeddingCache,
    *,
    weights: Dict[str, float] = FIELD_WEIGHTS,
) -> np.ndarray:
    """Weighted sum of normalized field embeddings, re-normalized."""
    vecs, wts = [], []
    for field, w in weights.items():
        text = meta.get(field)
        if not text or not isinstance(text, str):
            continue
        vecs.append(embedder.embed(text))
        wts.append(w)
    if not vecs:
        raise ValueError(f"Persona has no embeddable fields: {meta}")
    vecs = np.vstack(vecs)
    wts  = np.array(wts).reshape(-1, 1)
    emb  = (vecs * wts).sum(axis=0)
    return emb / (np.linalg.norm(emb) + 1e-12)


def _collect_unique_personas(
    edges: List[MutationEdge],
) -> tuple[list[str], dict[str, dict]]:
    """Return (ordered pid list, pid -> metadata dict) from edges."""
    meta: dict[str, dict] = {}
    order: list[str] = []
    for e in edges:
        pid = e.operator_metadata.get("persona_name")
        if pid and pid not in meta:
            meta[pid] = e.operator_metadata
            order.append(pid)
    return order, meta


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------

def cluster_personas_hdbscan(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    field_weights: Dict[str, float] = FIELD_WEIGHTS,
    vis_dir: str,
    min_cluster_size: int = 40,
    min_samples: Optional[int] = None,
) -> dict:
    """
    Cluster unique personas via HDBSCAN on weighted embeddings.

    Returns a dict with keys:
        persona_to_cluster  : pid -> cluster_id str
        cluster_members     : cluster_id -> [pid]
        cluster_summaries   : cluster_id -> summary dict
        cluster_stability   : cluster_id -> float
    """
    pids, persona_meta = _collect_unique_personas(edges)

    X = np.vstack([
        persona_to_weighted_embedding(persona_meta[pid], embedder, weights=field_weights)
        for pid in pids
    ])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(X)

    persona_to_cluster: dict[str, str] = {}
    cluster_members: dict[str, list]   = defaultdict(list)
    for pid, label in zip(pids, labels):
        cid = "persona_noise" if label == -1 else f"persona_cluster_{label}"
        persona_to_cluster[pid] = cid
        cluster_members[cid].append(pid)

    cluster_stability = {
        f"persona_cluster_{i}": float(s)
        for i, s in enumerate(clusterer.cluster_persistence_)
    }

    persona_embeddings = {pid: emb for pid, emb in zip(pids, X)}

    cluster_summaries = {
        cid: _summarize_cluster(persona_meta, persona_embeddings, members)
        for cid, members in cluster_members.items()
        if cid != "persona_noise"
    }

    plot_cluster_stability(
        cluster_members=dict(cluster_members),
        cluster_stability=cluster_stability,
        vis_dir=vis_dir,
    )

    return {
        "persona_to_cluster": persona_to_cluster,
        "cluster_members":    dict(cluster_members),
        "cluster_summaries":  cluster_summaries,
        "cluster_stability":  cluster_stability,
    }


def _summarize_cluster(
    persona_meta: dict,
    persona_embeddings: dict,
    cluster_pids: list,
    *,
    n_exemplars: int = 3,
) -> dict:
    X        = np.vstack([persona_embeddings[p] for p in cluster_pids])
    centroid = X.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sims     = X @ centroid
    sorted_i = np.argsort(sims)

    medoid_pid       = cluster_pids[int(np.argmax(sims))]
    peripheral_pid   = cluster_pids[sorted_i[0]]
    near_center_pid  = cluster_pids[sorted_i[len(sorted_i) // 2]]

    exemplar_pids = list(dict.fromkeys([medoid_pid, near_center_pid, peripheral_pid]))
    return {
        "size":                   len(cluster_pids),
        "representative_persona": medoid_pid,
        "exemplars":              [persona_meta[p] for p in exemplar_pids],
    }


# ---------------------------------------------------------------------------
# Relabelling edges
# ---------------------------------------------------------------------------

def relabel_edges_with_clusters(
    edges: List[MutationEdge],
    persona_to_cluster: dict,
) -> List[MutationEdge]:
    return [
        MutationEdge(
            **{
                **e.__dict__,
                "operator_name": persona_to_cluster.get(
                    e.operator_metadata.get("persona_name", ""),
                    "persona_noise",
                ),
            }
        )
        for e in edges
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cluster_stability(
    cluster_members: dict,
    cluster_stability: dict,
    vis_dir: str,
):
    ids, sizes, stabs, noise_size = [], [], [], 0
    for cid, members in cluster_members.items():
        if cid == "persona_noise":
            noise_size = len(members)
            continue
        if cid not in cluster_stability:
            continue
        ids.append(cid)
        sizes.append(len(members))
        stabs.append(cluster_stability[cid])

    plt.figure(figsize=(20, 14))
    plt.scatter(sizes, stabs, s=1000, color="green")
    for cid, x, y in zip(ids, sizes, stabs):
        plt.text(x + 0.5, y + 0.005, cid.replace("persona_cluster_", ""), fontsize=24)
    if noise_size > 0:
        plt.scatter([noise_size], [0.0], s=1000, color="blue")
        plt.text(noise_size + 0.5, 0.005, "noise", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Cluster size (# personas)", fontsize=24)
    plt.ylabel("Cluster stability", fontsize=24)
    plt.title("Persona Cluster Stability (HDBSCAN)", fontsize=26)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_cluster_stability.png"))
    plt.close()


def generate_persona_umap(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    cluster_summaries: dict,
    *,
    weights: Dict[str, float] = FIELD_WEIGHTS,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame:
    pids, persona_meta = _collect_unique_personas(edges)
    X = np.vstack([
        persona_to_weighted_embedding(persona_meta[p], embedder, weights=weights)
        for p in pids
    ])

    reducer    = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                           metric="cosine", random_state=42)
    embedding  = reducer.fit_transform(X)

    df         = pd.DataFrame(embedding, columns=["x", "y"])
    df["pid"]  = pids

    medoid_pids      = {s.get("representative_persona") for s in cluster_summaries.values()}
    df["is_medoid"]  = df["pid"].isin(medoid_pids)
    return df


def plot_persona_umap(umap_df: pd.DataFrame, vis_dir: str):
    plt.figure(figsize=(20, 16), dpi=300)
    sns.set_style("whitegrid")

    sns.scatterplot(data=umap_df[~umap_df["is_medoid"]], x="x", y="y",
                    alpha=0.4, s=80, color="gray", label="Persona mutations")

    medoids = umap_df[umap_df["is_medoid"]]
    plt.scatter(medoids["x"], medoids["y"], marker="*", s=250,
                c="red", edgecolor="black", label="Cluster centers", zorder=5)

    for _, row in medoids.iterrows():
        name = str(row["pid"])
        display = "\n".join(name[i:i+10] for i in range(0, len(name), 10))
        plt.annotate(display, (row["x"], row["y"]),
                     textcoords="offset points", xytext=(0, 20),
                     ha="center", va="bottom", fontsize=14, fontweight="bold",
                     linespacing=1.2,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    plt.title("Persona Embedding Space (UMAP)", fontsize=24, pad=25)
    plt.xlabel("UMAP Dimension 1", fontsize=22)
    plt.ylabel("UMAP Dimension 2", fontsize=22)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_umap.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_cluster_artifacts(
    cluster_members: dict,
    cluster_summaries: dict,
    vis_dir: str,
):
    with open(os.path.join(vis_dir, "persona_clusters.json"), "w") as f:
        json.dump(cluster_members, f, indent=2)
    with open(os.path.join(vis_dir, "persona_cluster_summaries.json"), "w") as f:
        json.dump(cluster_summaries, f, indent=2)
    print(f"[CLUSTERING] Saved cluster artifacts to {vis_dir}/")