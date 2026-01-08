from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable, Union, Tuple
import json
from collections import defaultdict, Counter
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

DISTANCE_THRESHOLD = 0.35 # cosine distance

FIELD_WEIGHTS = {
    "persona_obfuscation_strategy": 3.0,   # primary signal
    "persona_domain": 2.5,                 # secondary
    "persona_style": 1.0,
    "persona_tone": 0.5,
    "persona_name": 0.2            # almost negligible
}

def build_transform_color_map(transform_names):
    palette = sns.color_palette("tab10", n_colors=len(transform_names)+1)
    return {name: palette[i] for i, name in enumerate(transform_names)} | {"Average": palette[-1]}

@dataclass(frozen=True)
class MutationEdge:
    # --------
    # identity
    # --------
    iteration_id: int
    refinement_iter: int
    depth: int

    # -------------
    # root context
    # -------------
    root_prompt: str
    root_fitness: float
    source_dataset: Optional[str]

    # ----------------
    # mutation context
    # ----------------
    parent_prompt: str
    parent_fitness: float

    child_prompt: str
    child_fitness: float
    delta_fitness: float
    success: bool

    # ------------------
    # operator metadata
    # ------------------
    operator_type: str              # e.g. "transformation", "persona"
    operator_name: str              # transformation name
    operator_metadata: Dict[str, Any]

    # ------------------
    # convenience flags
    # ------------------
    is_root_edge: bool              # True if parent_prompt == root_prompt

def load_persona_edges_from_archive(
    archive_path: str,
    *,
    strict: bool = False,
) -> tuple[list[MutationEdge], int, int]:
    """
    Load persona-based red-teaming archive JSON and extract MutationEdge objects.
    Persona clustering is NOT done here.
    """
    with open(archive_path, "r") as f:
        archive = json.load(f)

    edges: list[MutationEdge] = []
    persona_counter = 0

    number_iterations = len(archive.items())
    print(f"[METRICS] Loading persona edges from archive: {archive_path} ({number_iterations} iterations)")

    for iter_key, iter_data in archive.items():
        try:
            iteration_id = int(iter_key)
        except ValueError:
            if strict:
                raise
            continue

        root_prompt = iter_data.get("parent_prompt")
        root_fitness = float(iter_data.get("parent_fitness", 0.0))

        parent_metadata = iter_data.get("parent_metadata", {})
        source_dataset = parent_metadata.get("source")

        attempts = iter_data.get("attempts", [])
        if not isinstance(attempts, list):
            continue

        for attempt in attempts:
            try:
                refinement_iter = int(attempt.get("refinement_iter", 0))
                depth = refinement_iter

                parent_prompt = attempt.get("parent_prompt")
                parent_fitness = float(attempt.get("parent_score", 0.0))

                persona = attempt.get("persona", {})
                persona_counter += 1

                persona_name = persona.get("name", "unknown")
                persona_style = persona.get("style", "")
                persona_tone = persona.get("tone", "")
                persona_domain = persona.get("domain", "")
                persona_obfuscation_strategy = persona.get("obfuscation_strategy", "")

                mutated_prompts = attempt.get("mutated_prompts", [])
                fitness = attempt.get("fitness", [])
                delta_fitness = attempt.get("delta_fitness", [])
                success_flags = attempt.get("success", [])

                n = min(
                    len(mutated_prompts),
                    len(fitness),
                    len(delta_fitness),
                    len(success_flags),
                )
                if n == 0:
                    continue

                for i in range(n):
                    edges.append(
                        MutationEdge(
                            iteration_id=iteration_id,
                            refinement_iter=refinement_iter,
                            depth=depth,

                            root_prompt=root_prompt,
                            root_fitness=root_fitness,
                            source_dataset=source_dataset,

                            parent_prompt=parent_prompt,
                            parent_fitness=parent_fitness,

                            child_prompt=mutated_prompts[i],
                            child_fitness=float(fitness[i]),
                            delta_fitness=float(delta_fitness[i]),
                            success=bool(success_flags[i]),

                            operator_type="persona",
                            operator_name=persona_name,  # TEMPORARY
                            operator_metadata={
                                "persona_name": persona_name,
                                "persona_style": persona_style,
                                "persona_tone": persona_tone,
                                "persona_domain": persona_domain,
                                "persona_obfuscation_strategy": persona_obfuscation_strategy,
                            },

                            is_root_edge=(parent_prompt == root_prompt),
                        )
                    )

            except Exception:
                if strict:
                    raise
                continue

    return edges, number_iterations, persona_counter

def load_mutation_edges_from_archives(
    archive_paths: Union[str, Iterable[str]],
    *,
    operator_type: str = "persona",
    strict: bool = False,
) -> list[MutationEdge]:
    """
    Load and merge multiple red-teaming archive JSONs into a single edge list.

    Args:
        archive_paths:
            - list of paths
            - directory
            - glob pattern (e.g. results/*.json)
        operator_type:
            "transformation" or "persona"
        strict:
            fail fast on malformed archives
        deduplicate:
            remove identical mutation edges across runs

    Returns:
        Flat list of MutationEdge with run_id populated
    """
    # ---------------------
    # resolve archive paths
    # ---------------------
    if isinstance(archive_paths, str):
        p = Path(archive_paths)
        if p.is_dir():
            paths = sorted(p.glob("*.json"))
        else:
            paths = sorted(Path().glob(archive_paths))
    else:
        paths = [Path(p) for p in archive_paths]

    all_edges: list[MutationEdge] = []
    all_persona_count = 0
    all_iteration_count = 0

    for path in paths:
        run_id = path.stem  # stable, human-readable

        edges, iteration_count, persona_count = load_persona_edges_from_archive(
            archive_path=str(path),
            strict=strict,
        )

        all_edges.extend(edges)
        all_persona_count += persona_count
        all_iteration_count += iteration_count
    
    print(f"[METRICS] Loaded {len(all_edges)} edges from {len(paths)} archives ({all_iteration_count} iterations, {all_persona_count} personas = {all_persona_count / all_iteration_count:.2f} personas/iteration)")

    return all_edges

def persona_to_text(meta: dict) -> str:
    return f"{meta.get('persona_name','')} | {meta.get('persona_style','')} | {meta.get('persona_domain','')}"

def persona_to_weighted_embedding(
    meta: dict,
    embedder,
    *,
    weights: dict,
) -> np.ndarray:
    """
    Compute a weighted persona embedding from structured fields.
    """
    vecs = []
    wts = []

    for field, w in weights.items():
        text = meta.get(field)
        if not text or not isinstance(text, str):
            continue

        v = embedder.embed(text)
        vecs.append(v)
        wts.append(w)

    if not vecs:
        raise ValueError("Persona has no embeddable fields")

    vecs = np.vstack(vecs)
    wts = np.array(wts).reshape(-1, 1)

    emb = (vecs * wts).sum(axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def cluster_personas_hdbscan(
    edges: list[MutationEdge],
    embedder,
    *,
    field_weights: dict,
    vis_dir: str,
    min_cluster_size: int = 40,
    min_samples: int | None = None,
):
    """
    Cluster personas using HDBSCAN.

    Returns:
    - persona_pid -> cluster_id
    - cluster_id -> list of persona pids
    """
    # -------------------------
    # collect unique personas
    # -------------------------
    persona_meta = {}
    persona_ids = []
    for e in edges:
        meta = e.operator_metadata
        pid = meta.get("persona_name")
        if pid and pid not in persona_meta:
            persona_meta[pid] = meta
            persona_ids.append(pid)

    pids = list(persona_meta.keys())

    # -------------------------
    # embed personas
    # -------------------------
    X = np.vstack([
        persona_to_weighted_embedding(
            persona_meta[pid],
            embedder,
            weights=field_weights,
        )
        for pid in pids
    ])

    # -------------------------
    # HDBSCAN clustering
    # -------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
    )

    labels = clusterer.fit_predict(X)

    # -------------------------
    # build mappings
    # -------------------------
    persona_to_cluster = {}
    cluster_members = defaultdict(list)

    for pid, label in zip(pids, labels):
        if label == -1:
            cid = "persona_noise"
        else:
            cid = f"persona_cluster_{label}"

        persona_to_cluster[pid] = cid
        cluster_members[cid].append(pid)
    
    cluster_stability = extract_cluster_stabilities(clusterer)

    plot_cluster_stability(
        cluster_members=cluster_members,
        cluster_stability=cluster_stability,
        vis_dir=vis_dir,
    )

    persona_embeddings = {
        pid: emb
        for pid, emb in zip(pids, X)
    }

    cluster_summaries = {}

    for cid, pids_in_cluster in cluster_members.items():
        if cid == "persona_noise":
            continue

        cluster_summaries[cid] = summarize_persona_cluster(
            persona_meta=persona_meta,
            persona_embeddings=persona_embeddings,
            cluster_pids=pids_in_cluster,
        )

    return {
        "persona_to_cluster": persona_to_cluster,
        "cluster_members": dict(cluster_members),
        "cluster_summaries": cluster_summaries,
        "cluster_stability": cluster_stability,
    }


def extract_cluster_stabilities(clusterer):
    """
    Returns:
    - cluster_id (string) -> stability (float)
    """
    stabilities = {}

    for cid, stability in enumerate(clusterer.cluster_persistence_):
        stabilities[f"persona_cluster_{cid}"] = float(stability)

    return stabilities


def plot_cluster_stability(
    cluster_members: dict,
    cluster_stability: dict,
    vis_dir: str,
):
    cluster_ids = []
    sizes = []
    stabilities = []

    noise_size = 0

    for cid, members in cluster_members.items():
        if cid == "persona_noise":
            noise_size = len(members)
            continue
        if cid not in cluster_stability:
            continue

        cluster_ids.append(cid)
        sizes.append(len(members))
        stabilities.append(cluster_stability[cid])

    sizes = np.array(sizes)
    stabilities = np.array(stabilities)



    plt.figure(figsize=(20, 14))
    plt.scatter(sizes, stabilities, s=1000, color="green")

    for cid, x, y in zip(cluster_ids, sizes, stabilities):
        plt.text(x+0.5, y+0.005, cid.replace("persona_cluster_", ""), fontsize=24)
    
    # plot noise cluster (if present)
    if noise_size > 0:
        plt.scatter([noise_size], [0.0], s=1000, color="blue")
        plt.text(noise_size+0.5, 0.005, "noise", fontsize=24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel("Cluster size (# personas)", fontsize=24)
    plt.ylabel("Cluster stability", fontsize=24)
    plt.title("Persona cluster stability (HDBSCAN)", fontsize=26)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_cluster_stability.png"))
    plt.close()


def summarize_persona_cluster(
    persona_meta: dict,
    persona_embeddings: dict,
    cluster_pids: list[str],
    *,
    top_k_fields: int = 3,
    n_exemplars: int = 3,
):
    """
    Returns a structured cluster summary, including:
    - cluster size
    - representative medoid persona
    - top modes of style, tone, domain, and obfuscation
    - exemplar personas: medoid, near-centroid, peripheral
    """
    if not cluster_pids:
        return {}

    # -------------------------
    # embeddings for the cluster
    # -------------------------
    X = np.vstack([persona_embeddings[pid] for pid in cluster_pids])
    centroid = X.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12

    # -------------------------
    # medoid (closest to centroid)
    # -------------------------
    sims = X @ centroid
    medoid_idx = int(np.argmax(sims))
    medoid_pid = cluster_pids[medoid_idx]
    medoid = persona_meta[medoid_pid]

    # -------------------------
    # peripheral & near-centroid exemplars
    # -------------------------
    sorted_indices = np.argsort(sims)  # ascending similarity
    peripheral_idx = sorted_indices[0]  # lowest similarity
    near_centroid_idx = sorted_indices[len(sorted_indices)//2]  # middle

    exemplar_pids = [medoid_pid, cluster_pids[near_centroid_idx], cluster_pids[peripheral_idx]]
    exemplar_pids = list(dict.fromkeys(exemplar_pids))  # remove duplicates if small cluster
    exemplar_personas = [persona_meta[pid] for pid in exemplar_pids]

    # -------------------------
    # aggregate categorical fields (modes)
    # -------------------------
    def top_values(field):
        vals = [
            persona_meta[pid].get(field)
            for pid in cluster_pids
            if persona_meta[pid].get(field)
        ]
        return [v for v, _ in Counter(vals).most_common(top_k_fields)]

    summary = {
        "size": len(cluster_pids),
        "representative_persona": medoid_pid,
        #"style_modes": top_values("persona_style"),
        #"tone_modes": top_values("persona_tone"),
        #"domain_modes": top_values("persona_domain"),
        #"obfuscation_modes": top_values("persona_obfuscation_strategy"),
        "exemplars": exemplar_personas
    }

    return summary

def cluster_personas_adaptive(
    edges: list[MutationEdge],
    embedder: SentenceTransformer,
    *,
    distance_threshold: float = 0.35,
):
    """
    Cluster personas adaptively using cosine distance threshold.

    Returns:
    - persona_name -> cluster_id
    - cluster_id -> list of persona names
    """

    # -------------------------
    # collect unique personas
    # -------------------------
    persona_meta = {}
    for e in edges:
        meta = e.operator_metadata
        name = meta.get("persona_name")
        if name and name not in persona_meta:
            persona_meta[name] = meta

    names = list(persona_meta.keys())
    if not names:
        return {}, {}

    # -------------------------
    # embed personas
    # -------------------------
    X = np.vstack([
        persona_to_weighted_embedding(persona_meta[name], embedder, weights=FIELD_WEIGHTS)
        for name in names
    ])

    # -------------------------
    # clustering (adaptive)
    # -------------------------
    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )

    labels = clustering.fit_predict(X)

    persona_to_cluster = {
        name: f"persona_cluster_{label}"
        for name, label in zip(names, labels)
    }

    cluster_members = defaultdict(list)
    for name, label in zip(names, labels):
        cluster_members[f"persona_cluster_{label}"].append(name)

    return persona_to_cluster, dict(cluster_members)


def cluster_personas(
    edges: list[MutationEdge],
    embedder: SentenceTransformer,
    *,
    n_clusters: int = 8,
):
    """
    Cluster personas and return:
    - persona_name -> cluster_id
    - cluster_id -> list of persona names
    """

    # -------------------------
    # collect unique personas
    # -------------------------
    persona_texts = {}
    for e in edges:
        meta = e.operator_metadata
        name = meta.get("persona_name")
        if name not in persona_texts:
            persona_texts[name] = persona_to_text(meta)

    names = list(persona_texts.keys())
    texts = [persona_texts[n] for n in names]

    # -------------------------
    # embed + cluster
    # -------------------------
    X = embedder.encode(texts, normalize_embeddings=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)

    persona_to_cluster = {
        name: f"persona_cluster_{label}"
        for name, label in zip(names, labels)
    }

    cluster_members = defaultdict(list)
    for name, label in zip(names, labels):
        cluster_members[f"persona_cluster_{label}"].append(name)

    return persona_to_cluster, dict(cluster_members)


def relabel_edges_with_persona_clusters(
    edges: list[MutationEdge],
    persona_to_cluster: dict,
):
    new_edges = []

    for e in edges:
        cluster_id = persona_to_cluster.get(
            e.operator_name,
            "persona_cluster_unknown"
        )

        new_edges.append(
            MutationEdge(
                **{**e.__dict__, "operator_name": cluster_id}
            )
        )

    return new_edges

def relabel_edges_with_persona_clusters_hdbscan(
    edges: list[MutationEdge],
    persona_to_cluster: dict,
):
    new_edges = []

    for e in edges:
        pid = e.operator_metadata.get("_pid") or e.operator_name
        cluster_id = persona_to_cluster.get(pid, "persona_noise")

        new_edges.append(
            MutationEdge(
                **{**e.__dict__, "operator_name": cluster_id}
            )
        )

    return new_edges

def generate_persona_umap(
    edges: List[Any], # List of MutationEdge objects
    embedder,
    cluster_summaries: Dict[int, dict],
    weights: Dict[str, float],
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> pd.DataFrame:
    """
    Extracts unique personas from MutationEdges, computes weighted embeddings,
    and prepares a DataFrame for plotting.
    """
    # 1. Deduplicate personas by their name/ID
    persona_meta = {}
    persona_ids = []
    for e in edges:
        meta = e.operator_metadata
        pid = meta.get("persona_name")
        if pid and pid not in persona_meta:
            persona_meta[pid] = meta
            persona_ids.append(pid)

    print(f"[UMAP] Processing {len(persona_meta)} unique personas.")

    # 2. Generate Embeddings
    X = np.vstack([
        persona_to_weighted_embedding(
            persona_meta[pid],
            embedder,
            weights=weights,
        )
        for pid in persona_ids
    ])
    print("[UMAP] Embedded personas")

    # 3. Dimensionality Reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(X)
    print("[UMAP] Reduced dimensions to 2D")

    # 4. Build DataFrame
    df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    df['pid'] = persona_ids
    print("[UMAP] Created DataFrame for visualization")
    
    # 5. Mark Medoids based on Cluster Summaries
    medoid_pids = {s.get("representative_persona") for s in cluster_summaries.values()}
    df['is_medoid'] = df['pid'].apply(lambda x: x in medoid_pids)
    print("[UMAP] Marked medoids in DataFrame")

    return df

def plot_embeddings(
    umap_df: pd.DataFrame,
    vis_dir: str,
):
    """
    Plots the UMAP projection with highlighted medoids, 
    consistent with large-scale high-resolution requirements.
    """
    plt.figure(figsize=(20, 16), dpi=300)
    sns.set_style("whitegrid")
    
    # Plot all personas (Mutations)
    sns.scatterplot(
        data=umap_df[~umap_df['is_medoid']],
        x='x', y='y',
        alpha=0.4,
        s=80,
        color='gray',
        label='Mutated Personas'
    )
    
    # Plot Medoids (The "Stars" of the clusters)
    medoids = umap_df[umap_df['is_medoid']]
    plt.scatter(
        medoids['x'], medoids['y'],
        marker='*',
        s=250,
        c='red',
        edgecolor='black',
        label='Cluster Centers',
        zorder=5
    )

# Annotate medoids with their names/IDs using newlines
    for _, row in medoids.iterrows():
        raw_name = str(row['pid'])
        
        # Example: insert newline every 10 chars
        display_name = "\n".join([raw_name[i:i+10] for i in range(0, len(raw_name), 10)])
        
        plt.annotate(
            display_name, 
            (row['x'], row['y']),
            textcoords="offset points",
            xytext=(0, 20), # Slightly higher to accommodate multiple lines
            ha='center',
            va='bottom',    # Vertical alignment bottom so it grows upwards
            fontsize=14,
            fontweight='bold',
            linespacing=1.2, # Adds a bit of breathing room between lines
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )

    plt.title("Personas Embedding Space", fontsize=24, pad=25)
    plt.xlabel("UMAP Dimension 1", fontsize=22)
    plt.ylabel("UMAP Dimension 2", fontsize=22)
    plt.legend(loc='best', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_umap.png"))
    plt.close()

def group_edges_by_operator(
    edges: List[MutationEdge],
) -> Dict[str, List[MutationEdge]]:
    grouped: Dict[str, List[MutationEdge]] = defaultdict(list)
    for e in edges:
        grouped[e.operator_name].append(e)
    return grouped

def group_edges_by_depth(edges: List[MutationEdge]) -> Dict[int, List[MutationEdge]]:
    by_depth = defaultdict(list)
    for e in edges:
        by_depth[e.depth].append(e)
    return dict(by_depth)

def group_edges_by_root(edges):
    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)
    return by_root

class EmbeddingCache:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self.cache[text]

    def cosine(self, a: str, b: str) -> float:
        va = self.embed(a)
        vb = self.embed(b)
        return float(np.dot(va, vb))

def plot_cumulative_metrics_vs_depth(
    depth_stats: dict,
    vis_dir: str,
):
    depths = sorted(depth_stats.keys())

    asr_vals = [depth_stats[d]["asr"] for d in depths]
    pres_vals = [depth_stats[d]["preservation"] for d in depths]
    jco_vals = [depth_stats[d]["jco"] for d in depths]
    print(asr_vals)
    print(pres_vals)
    print(jco_vals)

    plt.figure(figsize=(12, 8))

    plt.plot(depths, asr_vals, marker="o", linewidth=3, label="Cumulative ASR")
    plt.plot(depths, pres_vals, marker="s", linewidth=3, label="Cumulative Semantic Preservation")
    plt.plot(depths, jco_vals, marker="^", linewidth=3, label="Cumulative JCO")

    plt.xlabel("Refinement Depth", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.title("Cumulative Attack Metrics vs Refinement Depth", fontsize=24)

    plt.xticks(depths, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "cumulative_metrics_vs_depth.png"),
        dpi=200,
    )
    plt.close()


def plot_metrics_vs_depth(
    depth_stats: Dict[int, Dict[str, float]],
    vis_dir: str,
):
    depths = sorted(depth_stats.keys())

    asr_vals = [depth_stats[d]["asr"] for d in depths]
    pres_vals = [depth_stats[d]["preservation"] for d in depths]
    jco_vals = [depth_stats[d]["jco"] for d in depths]

    plt.figure(figsize=(12, 8))

    plt.plot(
        depths, asr_vals,
        marker="o",
        linewidth=3,
        label="Attack Success Rate (ASR)",
    )
    plt.plot(
        depths, pres_vals,
        marker="s",
        linewidth=3,
        label="Semantic Preservation (SP)",
    )
    plt.plot(
        depths, jco_vals,
        marker="^",
        linewidth=3,
        label="Judge-Consistent Obfuscation (JCO)",
    )

    plt.xlabel("Refinement Depth", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.title("Evolution of Metrics Across Refinement Depth", fontsize=24)

    plt.xticks(depths, fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "metrics_vs_depth.png"),
        dpi=200,
    )
    plt.close()


def cumulative_asr_by_depth(edges):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    asr = {}
    for d in range(max_depth + 1):
        successes = 0
        for root, es in by_root.items():
            if any(e.success and e.depth <= d for e in es):
                successes += 1
        asr[d] = successes / len(by_root)

    return asr


def cumulative_jco_by_depth(edges, embedder):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    jco = {}
    for d in range(max_depth + 1):
        values = []
        for root, es in by_root.items():
            best = 0.0
            for e in es:
                if e.depth <= d and e.success:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    best = max(best, sim)
                    values.append(sim)
                #values.append(best)
        jco[d] = float(np.mean(values))
    jco[0] = 0.0  # at depth 0, JCO is 0.0
    return jco


def cumulative_preservation_by_depth(edges, embedder):
    by_root = group_edges_by_root(edges)
    max_depth = max(e.depth for e in edges)

    preservation = {}
    for d in range(max_depth + 1):
        values = []
        """
        for root, es in by_root.items():
            best = 0.0
            for e in es:
                if e.depth == d:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    best = max(best, sim)
            values.append(best)
        preservation[d] = float(np.mean(values))
        """
        for root, es in by_root.items():
            for e in es:
                if e.depth <= d:
                    sim = embedder.cosine(e.root_prompt, e.child_prompt)
                    values.append(sim)
        preservation[d] = float(np.mean(values))
    preservation[0] = 1.0  # at depth 0, preservation is perfect

    return preservation


def cumulative_metrics_by_depth(edges, embedder):
    asr = cumulative_asr_by_depth(edges)
    jco = cumulative_jco_by_depth(edges, embedder)
    preservation = cumulative_preservation_by_depth(edges, embedder)

    depths = sorted(asr.keys())
    stats = {}

    for d in depths:
        stats[d] = {
            "asr": asr[d],
            "jco": jco[d],
            "preservation": preservation[d],
        }

    return stats


def global_semantic_preservation(edges, embedder, reference="root"):
    preservations = semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)) if preservations else 0.0,
        "std": float(np.std(preservations)) if preservations else 0.0,
    }

def global_root_level_semantic_preservation(edges, embedder, reference="root"):
    preservations = root_level_semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)) if preservations else 0.0,
        "std": float(np.std(preservations)) if preservations else 0.0,
    }


def global_depth_of_success(edges):
    depths = [e.depth for e in edges if e.success]
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max": int(max(depths)) if depths else 0,
    }

def global_root_level_depth_of_success(edges):
    by_root = group_edges_by_root(edges)
    depths = []
    for root, es in by_root.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append(min(successful_depths))
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max": int(max(depths)) if depths else 0,
    }

def global_root_level_avg_depth_of_success(edges):
    by_root = group_edges_by_root(edges)
    depths = []
    for root, es in by_root.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append(np.mean(successful_depths))
    return {
        "mean": float(np.mean(depths)) if depths else 0.0,
        "max": int(max(depths)) if depths else 0,
    }


def collect_results(
    edges,
    grouped_edges,
    embedder,
):
    results = {}
    unique_root_set = {e.root_prompt for e in edges}

    # ----------------
    # Global statistics
    # ----------------
    results["global"] = {
        "num_edges": len(edges),
        "num_total_mutations": len(edges),
        "num_unique_root_prompts": len(unique_root_set),
        "num_personas": len(edges)/3, # batch size of 3 personas per mutations
        "edge_level_attack_success_rate": attack_success_rate(edges),
        "edge_level_semantic_preservation": global_semantic_preservation(edges, embedder),
        "edge_level_judge_consistent_obfuscation": judge_consistent_obfuscation(edges, embedder),
        "edge_level_depth_of_success": global_depth_of_success(edges),
        "root_level_attack_success_rate": root_level_attack_success_rate(edges),
        "root_level_semantic_preservation": global_root_level_semantic_preservation(edges, embedder),
        "root_level_judge_consistent_obfuscation": root_level_jco(edges, embedder),
        "root_level_min_depth_of_success": global_root_level_depth_of_success(edges),
        "root_level_avg_depth_of_success": global_root_level_avg_depth_of_success(edges),
        "mutation_efficiency_index": mutation_efficiency_index(edges, embedder),
        "semantic_drift_rate": compute_semantic_drift_rate(edges, embedder)
    }

    # -------------------------
    # Per-transformation stats
    # -------------------------
    if grouped_edges is None:
        return results
    per_transform = {}

    asr_per = attack_success_rate_per_transform(grouped_edges)
    root_asr_per = root_level_asr_per_transform(grouped_edges)
    preservation_per = semantic_preservation_per_transform(grouped_edges, embedder)
    root_preservation_per = root_level_semantic_preservation_per_transform(grouped_edges, embedder)
    jco_per = jco_per_transform(grouped_edges, embedder)
    root_jco_per = root_level_jco_per_transform(grouped_edges, embedder)
    depth_per = depth_of_success_per_transform(grouped_edges)
    root_depth_per = root_level_depth_of_success_per_transform(grouped_edges)
    mei = mei_per_transform(grouped_edges, embedder)
    semantic_drift = drift_rate_per_transform(grouped_edges, embedder)

    print("[METRICS] Grouped edges: ", {k: len(v) for k, v in grouped_edges.items()})
    for name, edges_t in grouped_edges.items():
        per_transform[name] = {
            "num_edges": len(edges_t),
            "global_attack_success_rate": asr_per[name],
            "root_level_attack_success_rate": root_asr_per[name],
            "global_semantic_preservation": preservation_per[name],
            "root_level_semantic_preservation": root_preservation_per[name],
            "global_judge_consistent_obfuscation": jco_per[name],
            "root_level_judge_consistent_obfuscation": root_jco_per[name],
            "global_depth_of_success": depth_per[name],
            "root_level_depth_of_success": root_depth_per[name],
            "mutation_efficiency_index": mei[name],
            "semantic_drift_rate": semantic_drift[name],
        }

    results["per_transformation"] = per_transform

    return results

def save_results_json(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[RESULTS] Saved metrics to {output_path}")

# -------------------------
# ------- METRICS ---------
# -------------------------
# -- Attack Success Rate -- 
def attack_success_rate(
    edges: List[MutationEdge],
) -> float:
    if not edges:
        return 0.0
    return np.mean([e.success for e in edges])

# -- Attack Success Rate per Transformation --
def attack_success_rate_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {
        name: attack_success_rate(edges)
        for name, edges in grouped_edges.items()
    }

def plot_asr(asr_per_transform: Dict[str, float], vis_dir: str, color_map):
    names = list(asr_per_transform.keys())
    values = list(asr_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)
    plt.ylabel("ASR_G", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.grid(axis="y", alpha=0.3)
    plt.yticks(fontsize=18)
    plt.title("Global Attack Success Rate per Transformation (ASR_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_attack_success_rate_per_transform.png"))
    plt.close()

# -- Root-Level Attack Success Rate --
def root_level_attack_success_rate(
    edges: List[MutationEdge],
) -> float:
    if not edges:
        return 0.0

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)

    successes = 0
    for root, es in by_root.items():
        if any(e.success for e in es): # counting if at least one successful mutation per root
            successes += 1

    return successes / len(by_root)

# -- Root-Level ASR per Transformation --
def root_level_asr_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
) -> Dict[str, float]:
    return {
        name: root_level_attack_success_rate(edges)
        for name, edges in grouped_edges.items()
    }

# -- Plot Root-Level ASR --
def plot_root_level_asr(
    root_asr_per_transform: Dict[str, float],
    vis_dir: str,
    color_map,
):
    names = list(root_asr_per_transform.keys())
    values = list(root_asr_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)

    plt.ylabel("ASR_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)

    plt.title(
        "Root-level Attack Success Rate per Transformation (ASR_R)",
        fontsize=24,
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(vis_dir, "root_level_attack_success_rate_per_transform.png")
    )
    plt.close()

# -- Semantic Preservation --
def semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",  # "parent" or "root"
) -> List[float]:
    preservations = []

    for e in edges:
        ref_text = e.parent_prompt if reference == "parent" else e.root_prompt
        sim = embedder.cosine(ref_text, e.child_prompt)
        preservations.append(sim)

    return preservations

# -- Semantic Preservation Stats --
def semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
):
    preservations = semantic_preservation(edges, embedder)
    return {
        "mean": float(np.mean(preservations)),
        "std": float(np.std(preservations)),
    }

# -- Semantic Preservation per Transformation --
def semantic_preservation_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: semantic_preservation_stats(edges, embedder)
        for name, edges in grouped_edges.items()
    }

# -- Semantic Preservation per Persona --
def semantic_preservation_per_persona(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
):
    """
    Returns a list of tuples (persona_name, mean_sp) for every instance.
    """
    results = []
    for e in edges:
        name = e.operator_metadata.get("persona_name", "unknown")
        stats = semantic_preservation_stats([e], embedder)
        results.append((name, stats["mean"]))
    
    # Sort by the mean value (index 1) in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def plot_semantic_preservation(preservation_per_transform: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(preservation_per_transform.keys())
    means = [v["mean"] for v in preservation_per_transform.values()]
    stds = [v["std"] for v in preservation_per_transform.values()]
    colors = [color_map[n] for n in names]

    mean = np.mean(means)
    names.append("AVERAGE")
    means.append(mean)
    stds.append(np.std(means))
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, yerr=stds, capsize=4, color=colors)
    plt.ylabel("SP_G", fontsize=22)
    plt.yticks(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.title("Global Semantic Preservation per Transformation (SP_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_semantic_preservation_per_transform.png"))
    plt.close()

# -- Semantic Preservation per Persona --    
def plot_semantic_preservation_persona(sorted_data, vis_dir: str, color_map=None):
    # Unpack the sorted tuples
    names = [item[0] for item in sorted_data]
    means = [item[1] for item in sorted_data]
    
    n_personas = len(names)
    avg_mean = np.mean(means) if means else 0.0
    x_pos = np.arange(n_personas)

    # Prepare colors based on the base persona name
    if color_map:
        colors = [color_map.get(n, "gray") for n in names]
    else:
        colors = ["#3498db"] * n_personas # A nice clean blue if no map is provided

    plt.figure(figsize=(20, 8))
    
    # linewidth=0 and edgecolor='none' are CRITICAL for 20k+ bars
    plt.bar(x_pos, means, color=colors, width=0.8, linewidth=0)
    
    # Plot the average line
    plt.axhline(avg_mean, color='red', linestyle='--', linewidth=2, label=f'AVERAGE SP_G ({avg_mean:.2f})')

    plt.ylabel("Global Semantic Preservation (SP_G)", fontsize=22)
    plt.xlabel("Personas, sorted by SP_G", fontsize=22)
    plt.xticks([]) # Hide X-ticks as 21,000 labels would be unreadable
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.1) # Similarity is usually 0-1
    
    plt.title("Distribution of Global Semantic Preservation across all Personas", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_edge_level_semantic_preservation_sorted.png"), dpi=300)
    plt.close()

def plot_semantic_preservation_violin(
    grouped_edges,
    embedder,
    vis_dir,
    color_map,
    reference="root",
):
    data = []
    labels = []

    for name, edges in grouped_edges.items():
        preservations = semantic_preservation(edges, embedder, reference=reference)
        data.append(preservations)
        labels.append(name)

    plt.figure(figsize=(24, 10))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_map[labels[i]])
        pc.set_alpha(0.7)

    plt.grid(axis="y", alpha=0.3)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=22)
    plt.ylabel("SP_G", fontsize=24)
    plt.yticks(fontsize=20)
    plt.title("Global Semantic Preservation Distribution per Transformation (SP_G)", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_semantic_preservation_violin.png"))
    plt.close()


# -- Root-Level Semantic Preservation --
def root_level_semantic_preservation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> List[float]: # Return a list, not a float
    if not edges:
        return []

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)

    values = []
    for root, es in by_root.items():
        val = []
        # To match your report: find the terminal mutation (highest depth)
        # regardless of success, or use your success logic if preferred.
        terminal_edges = [e for e in es if e.success or e.depth == max(ev.depth for ev in es)]
        
        for terminal_edge in terminal_edges:
            ref = terminal_edge.parent_prompt if reference == "parent" else terminal_edge.root_prompt
            sim = embedder.cosine(ref, terminal_edge.child_prompt)
            val.append(sim)
        values.append(np.mean(val))  # Average if multiple terminal edges per root

    return values

# -- Root-Level Semantic Preservation Stats --
def root_level_semantic_preservation_stats(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    preservations = root_level_semantic_preservation(edges, embedder, reference=reference)
    
    if not preservations:
        return {"mean": 0.0, "std": 0.0}
        
    return {
        "mean": float(np.mean(preservations)),
        "std": float(np.std(preservations)),
    }

# -- Root-Level Semantic Preservation per Transformation --
def root_level_semantic_preservation_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
):
    return {
        name: root_level_semantic_preservation_stats(edges, embedder, reference=reference)
        for name, edges in grouped_edges.items()
    }

# -- Root-Level Semantic Preservation per Persona --
def root_level_semantic_preservation_per_persona(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
):
    res = root_level_semantic_preservation(edges, embedder)
    results = []
    for e, r in zip(edges, res):
        name = e.operator_metadata.get("persona_name", "unknown")
        results.append((name, r))
    
    # Sort by the mean value (index 1) in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def plot_root_level_semantic_preservation(
    root_preservation_per_transform: Dict[str, Dict[str, float]], # Correct type hint
    vis_dir: str,
    color_map,
):
    names = list(root_preservation_per_transform.keys())
    means = [v["mean"] for v in root_preservation_per_transform.values()]
    stds = [v["std"] for v in root_preservation_per_transform.values()]
    
    # Ensure color_map key exists (using .get() or checking previous logic)
    colors = [color_map.get(n, "blue") for n in names]

    global_mean = np.mean(means) if means else 0.0
    names.append("AVERAGE")
    means.append(global_mean)
    stds.append(np.std(means) if len(means) > 1 else 0.0)
    
    # Use the same color logic for average as your other plots
    colors.append(color_map.get("Average", color_map.get(-1, "black")))

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, yerr=stds, capsize=4, color=colors)
    plt.ylabel("SP_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)
    plt.title("Root-level Semantic Preservation per Transformation (SP_R)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_semantic_preservation_per_transform.png"))
    plt.close()


def plot_root_level_semantic_preservation_persona(sorted_data, vis_dir: str, color_map=None):
    # Unpack the sorted tuples
    names = [item[0] for item in sorted_data]
    means = [item[1] for item in sorted_data]
    
    n_personas = len(names)
    avg_mean = np.mean(means) if means else 0.0
    x_pos = np.arange(n_personas)

    # Prepare colors based on the base persona name
    if color_map:
        colors = [color_map.get(n, "gray") for n in names]
    else:
        colors = ["#2d39bd"] * n_personas # A nice clean blue if no map is provided

    plt.figure(figsize=(20, 8))
    
    # linewidth=0 and edgecolor='none' are CRITICAL for 20k+ bars
    plt.bar(x_pos, means, color=colors, width=0.8, linewidth=0)
    
    # Plot the average line
    plt.axhline(avg_mean, color='red', linestyle='--', linewidth=2, label=f'AVERAGE SP_R ({avg_mean:.2f})')

    plt.ylabel("Root-level Semantic Preservation (SP_R)", fontsize=22)
    plt.xlabel("Personas, sorted by SP_R", fontsize=22)
    plt.xticks([]) # Hide X-ticks as 21,000 labels would be unreadable
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.1) # Similarity is usually 0-1
    
    plt.title("Distribution of Root-level Semantic Preservation across all Personas", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_root_level_semantic_preservation_sorted.png"), dpi=300)
    plt.close()


# -- Success vs Preservation Data --
def get_success_vs_preservation_data(edges: List[MutationEdge], embedder: EmbeddingCache):
    """
    Extracts success status and semantic preservation for every edge.
    """
    data = []
    for e in edges:
        # Calculate SP for this specific edge
        sp = embedder.cosine(e.root_prompt, e.child_prompt)
        data.append({
            "success": int(e.success),
            "sp": sp
        })
    return data

def plot_success_vs_semantic_preservation(edges, embedder, vis_dir):
    # 1. Extract Data
    raw_data = get_success_vs_preservation_data(edges, embedder)
    
    success_sp = [d["sp"] for d in raw_data if d["success"] == 1]
    failure_sp = [d["sp"] for d in raw_data if d["success"] == 0]
    
    # 2. Setup Figure
    plt.figure(figsize=(12, 10)) # Square-ish to show the vertical distribution clearly
    
    # 3. Plot Points with Jitter
    # Jitter spreads the points out horizontally so we can see density
    def get_jitter(val, length):
        return np.random.normal(val, 0.04, length)

    plt.scatter(get_jitter(0, len(failure_sp)), failure_sp, 
                color='red', alpha=0.1, s=10, label='Failure (0)', rasterized=True)
    
    plt.scatter(get_jitter(1, len(success_sp)), success_sp, 
                color='blue', alpha=0.1, s=10, label='Success (1)', rasterized=True)

    # 4. Plot Averages (The "Clusters")
    avg_fail = np.mean(failure_sp) if failure_sp else 0
    avg_succ = np.mean(success_sp) if success_sp else 0
    
    # Mean markers (Large white-bordered diamonds)
    plt.plot(0, avg_fail, marker='D', markersize=15, color='white', markeredgecolor='black', markeredgewidth=2)
    plt.plot(0, avg_fail, marker='D', markersize=12, color='red', label=f'Avg Failure ({avg_fail:.2f})')
    
    plt.plot(1, avg_succ, marker='D', markersize=15, color='white', markeredgecolor='black', markeredgewidth=2)
    plt.plot(1, avg_succ, marker='D', markersize=12, color='blue', label=f'Avg Success ({avg_succ:.2f})')

    # 5. Styling (Matching your font sizes)
    plt.title("Semantic Preservation vs. Attack Success", fontsize=24)
    plt.ylabel("Global Semantic Preservation ($SP_G$)", fontsize=22)
    plt.xlabel("Attack Success (0 = Fail, 1 = Success)", fontsize=22)
    
    plt.xticks([0, 1], ["Failure", "Success"], fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.1)
    plt.xlim(-0.5, 1.5)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=16, loc='lower center', ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "success_vs_preservation_scatter.png"), dpi=300)
    plt.close()


# -- Judge Consistent Obfuscation (JCO) --
def judge_consistent_obfuscation(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    scores = []
    for e in edges:
        sim = embedder.cosine(e.parent_prompt, e.child_prompt) if reference == "parent" else embedder.cosine(e.root_prompt, e.child_prompt) # JCO relative to root or parent
        jco = float(e.success) * sim
        scores.append(jco)
    return np.mean(scores) if scores else 0.0

# -- JCO per Transformation --
def jco_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: judge_consistent_obfuscation(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_jco(jco_scores: Dict[str, float], vis_dir: str, color_map):
    names = list(jco_scores.keys())
    values = list(jco_scores.values())
    colors = [color_map[n] for n in names]

    global_jco = np.mean(values)
    values.append(global_jco)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)
    plt.ylabel("JCO_G", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(np.arange(0, 0.5, 0.1), fontsize=18)
    plt.title("Global JCO per Transformation (JCO_G)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_jco_per_transform.png"))
    plt.close()


# -- Root-Level JCO --
def root_level_jco(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
) -> float:
    if not edges:
        return 0.0

    by_root = defaultdict(list)
    for e in edges:
        by_root[e.root_prompt].append(e)

    values = []
    for root, es in by_root.items():
        best = 0.0
        for e in es:
            if not e.success:
                continue
            ref = e.parent_prompt if reference == "parent" else e.root_prompt
            sim = embedder.cosine(ref, e.child_prompt)
            best = max(best, sim)
        values.append(best)

    return float(np.mean(values))


# -- Root-Level JCO per Transformation --
def root_level_jco_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
    *,
    reference: str = "root",
):
    return {
        name: root_level_jco(edges, embedder, reference=reference)
        for name, edges in grouped_edges.items()
    }


def plot_root_level_jco(
    root_jco_per_transform: Dict[str, float],
    vis_dir: str,
    color_map,
):
    names = list(root_jco_per_transform.keys())
    values = list(root_jco_per_transform.values())
    colors = [color_map[n] for n in names]

    global_asr = np.mean(values)
    values.append(global_asr)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)
    plt.ylabel("JCO_R", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(np.arange(0, 0.8, 0.1),fontsize=18)
    plt.title("Root-level JCO per Transformation (JCO_R)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_jco_per_transform.png"))
    plt.close()

# -- Mutation Efficiency Index (MEI) --
def mutation_efficiency_index(
    edges: List[MutationEdge],
    embedder: EmbeddingCache,
    reference: str = "root",
):
    scores = []
    for e in edges:
        sim = embedder.cosine(e.parent_prompt, e.child_prompt) if reference == "parent" else embedder.cosine(e.root_prompt, e.child_prompt) # JCO relative to root or parent
        mei = float(e.success) * sim / e.depth
        scores.append(mei)
    return np.mean(scores) if scores else 0.0

# -- MEI per Transformation --
def mei_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
    embedder: EmbeddingCache,
):
    return {
        name: mutation_efficiency_index(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_mei(mei_scores: Dict[str, float], vis_dir: str, color_map):
    names = list(mei_scores.keys())
    values = list(mei_scores.values())
    colors = [color_map[n] for n in names]

    global_mei = np.mean(values)
    values.append(global_mei)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, values, color=colors)
    plt.ylabel("MEI", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=18)
    plt.title("Mutation Efficiency Index per Transformation (MEI)", fontsize=24)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "mei_per_transform.png"))
    plt.close()


def plot_asr_preservation_heatmap(
    asr_per_transform,
    preservation_per_transform,
    vis_dir,
    *,
    type="edge", # "edge" or "root"
):
    sns.set_context("talk", font_scale=2.2)

    rows = []
    for name in asr_per_transform:
        rows.append([
            name,
            asr_per_transform[name],
            preservation_per_transform[name]["mean"],
        ])

    df = pd.DataFrame(
        rows,
        columns=["Transformation", "ASR", "Semantic Preservation"],
    ).set_index("Transformation")

    plt.figure(
        figsize=(16, max(6, 0.9 * len(df))),
        dpi=200,
    )

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 26},
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "ASR and Semantic Preservation per Transformation",
        fontsize=32,
        pad=25,
    )

    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(
        os.path.join(vis_dir, f"{type}_asr_preservation_heatmap.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_asr_vs_preservation(
    asr_per_transform,
    preservation_per_transform,
    vis_dir,
    color_map,
    *,
    type="edge", # "edge" or "root"
):
    plt.figure(figsize=(12, 10))

    for name in asr_per_transform:
        plt.scatter(
            preservation_per_transform[name]["mean"],
            asr_per_transform[name],
            color=color_map[name],
            label=name,
            s=300,
        )
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Semantic Preservation (⟶ better)", fontsize=20)
    plt.ylabel("Attack Success Rate (⟶ better)", fontsize=20)
    plt.title("ASR vs Semantic Preservation", fontsize=22)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{type}_asr_vs_preservation.png"))
    plt.close()

# -- Depth of Success --
def depth_of_success(edges: List[MutationEdge]) -> List[int]:
    depths = []
    for e in edges:
        if e.success:
            depths.append(e.depth)
    return depths

# -- Depth of Success per Transformation --
def depth_of_success_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
):
    stats = {}
    for name, edges in grouped_edges.items():
        depths = depth_of_success(edges)
        if depths:
            stats[name] = {
                "mean": np.mean(depths),
                "median": np.median(depths),
            }
    return stats

def plot_depth_of_success(depth_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(depth_stats.keys())
    means = [v["mean"] for v in depth_stats.values()]
    colors = [color_map[n] for n in names]

    global_depth = np.mean(means)
    means.append(global_depth)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, color=colors)
    plt.ylabel("d_G", fontsize=22)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks([0,1,2,3,4], fontsize=18)
    plt.title("Global Depth of Success per Transformation (d_G)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "edge_level_depth_of_success_per_transform.png"))
    plt.close()

# -- Root-Level Depth of Success --
def root_level_depth_of_success(edges: List[MutationEdge]) -> List[int]:
    depths = []
    root_edges = group_edges_by_root(edges)
    for root, es in root_edges.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append(min(successful_depths))
    return depths

# -- Root-Level Depth of Success per Transformation --
def root_level_depth_of_success_per_transform(
    grouped_edges: Dict[str, List[MutationEdge]],
):
    stats = {}
    for name, edges in grouped_edges.items():
        depths = root_level_depth_of_success(edges)
        if depths:
            stats[name] = {
                "mean": np.mean(depths),
                "median": np.median(depths),
            }
    return stats

def plot_root_level_depth_of_success(depth_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map):
    names = list(depth_stats.keys())
    means = [v["mean"] for v in depth_stats.values()]
    colors = [color_map[n] for n in names]

    global_depth = np.mean(means)
    means.append(global_depth)
    names.append("AVERAGE")
    colors.append(color_map["Average"])

    plt.figure(figsize=(20, 8))
    plt.bar(names, means, color=colors)
    plt.ylabel("d_R", fontsize=22)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(np.arange(0, 2.0, 0.5),fontsize=18)
    plt.title("Root-Level Depth of Success per Transformation (d_R)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_depth_of_success_per_transform.png"))
    plt.close()


# -- Edge-level depth of success personas --
def edge_level_depth_of_success_per_persona(edges: List[MutationEdge]):
    results = []
    for e in edges:
        name = e.operator_metadata.get("persona_name", "unknown")
        results.append((name, e.depth if e.success else 0))
    
    # Sort by the mean value (index 1) in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def plot_persona_edge_depth_of_success(sorted_data, vis_dir: str, color_map=None):
    # Unpack the sorted tuples (name, depth)
    names = [item[0] for item in sorted_data]
    depths = [item[1] for item in sorted_data]
    
    n_instances = len(names)
    # Average only includes successful ones (depth > 0)
    succ_depths = [d for d in depths if d > 0]
    avg_depth = np.mean(succ_depths) if succ_depths else 0.0
    
    x_pos = np.arange(n_instances)

    # Color mapping
    if color_map:
        colors = [color_map.get(n, "gray") for n in names]
    else:
        colors = ["#079D68"] * n_instances # A green color for success depth

    plt.figure(figsize=(20, 8))
    
    # Plotting the "Staircase" of depths
    plt.bar(x_pos, depths, color=colors, width=1.0, linewidth=0, edgecolor='none')
    
    # Plot the average line for successful mutations
    plt.axhline(avg_depth, color='red', linestyle='--', linewidth=2, 
                label=f'AVG Depth of Success ({avg_depth:.2f})')

    plt.ylabel("Refinement Depth", fontsize=22)
    plt.xlabel(f"Personas, sorted by depth", fontsize=22)
    plt.xticks([]) 
    plt.yticks(range(0, 6), fontsize=18)
    plt.ylim(0, 5.5) 
    
    plt.title("Distribution of Edge-level Depth of Success across all Personas", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_edge_depth_of_success_sorted.png"), dpi=300)
    plt.close()

# -- Root-level depth of success personas --
def root_level_min_depth_of_success_per_persona(edges: List[MutationEdge]):
    depths = []
    root_edges = group_edges_by_root(edges)
    for root, es in root_edges.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append((root, min(successful_depths)))
    return depths

def root_level_avg_depth_of_success_per_persona(edges: List[MutationEdge]):
    depths = []
    root_edges = group_edges_by_root(edges)
    for root, es in root_edges.items():
        successful_depths = [e.depth for e in es if e.success]
        if successful_depths:
            depths.append((root, np.mean(successful_depths)))
    return depths

def plot_root_level_min_depth_of_success(root_data, vis_dir: str):
    """
    root_data: List of tuples (root_prompt, min_depth)
    """
    # Sort by depth descending to maintain the "staircase" style
    sorted_roots = sorted(root_data, key=lambda x: x[1], reverse=True)
    
    names = [item[0] for item in sorted_roots]
    depths = [item[1] for item in sorted_roots]
    
    n_roots = len(names)
    avg_depth = np.mean(depths) if depths else 0.0
    x_pos = np.arange(n_roots)

    plt.figure(figsize=(20, 8))
    
    # Using a distinct color (Purple) for root-level metrics
    plt.bar(x_pos, depths, color="#9b59b6", width=0.8, linewidth=0)
    
    plt.axhline(avg_depth, color='red', linestyle='--', linewidth=2, 
                label=f'AVG Root Success Depth ({avg_depth:.2f})')

    plt.ylabel("Minimum Depth to Success", fontsize=22)
    plt.xlabel(f"Unique Root Prompts, sorted by minimum depth", fontsize=22)
    plt.xticks([]) 
    plt.yticks(range(0, 6), fontsize=18)
    plt.ylim(0, 5.5) 
    
    plt.title("Root-level Minimum Depth of Success", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_depth_of_success_sorted.png"), dpi=300)
    plt.close()

def plot_root_level_avg_depth_of_success(root_data, vis_dir: str):
    """
    root_data: List of tuples (root_prompt, avg_depth)
    """
    # Sort by depth descending to maintain the "staircase" style
    sorted_roots = sorted(root_data, key=lambda x: x[1], reverse=True)
    
    names = [item[0] for item in sorted_roots]
    depths = [item[1] for item in sorted_roots]
    
    n_roots = len(names)
    avg_depth = np.mean(depths) if depths else 0.0
    x_pos = np.arange(n_roots)

    plt.figure(figsize=(20, 8))
    
    # Using a distinct color (Purple) for root-level metrics
    plt.bar(x_pos, depths, color="#773a8f", width=0.8, linewidth=0)
    
    plt.axhline(avg_depth, color='red', linestyle='--', linewidth=2, 
                label=f'AVG Root Success Depth ({avg_depth:.2f})')

    plt.ylabel("Average Depth to Success", fontsize=22)
    plt.xlabel(f"Unique Root Prompts, sorted by average depth", fontsize=22)
    plt.xticks([]) 
    plt.yticks(range(0, 6), fontsize=18)
    plt.ylim(0, 5.5) 
    
    plt.title("Root-level Average Depth of Success", fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "root_level_avg_depth_of_success_sorted.png"), dpi=300)
    plt.close()


# -- Drift rate --
def compute_semantic_drift_rate(edges: List[MutationEdge], embedder: EmbeddingCache) -> Dict[str, float]:
    """
    Calculates the average semantic drift (loss of SP) per mutation step.
    """
    drifts = []
    
    for e in edges:
        # Similarity of Parent to Root (If depth 1, parent is root, so sim is 1.0)
        if e.depth <= 1:
            sim_parent_root = 1.0
        else:
            sim_parent_root = embedder.cosine(e.root_prompt, e.parent_prompt)
            
        # Similarity of Child to Root
        sim_child_root = embedder.cosine(e.root_prompt, e.child_prompt)
        
        # Drift = Loss in preservation
        drift = sim_parent_root - sim_child_root
        drifts.append(drift)
        
    return {
        "mean": float(np.mean(drifts)) if drifts else 0.0,
        "std": float(np.std(drifts)) if drifts else 0.0
    }

def drift_rate_per_transform(grouped_edges, embedder):
    return {
        name: compute_semantic_drift_rate(edges, embedder)
        for name, edges in grouped_edges.items()
    }

def plot_drift_rate(drift_stats: Dict[str, Dict[str, float]], vis_dir: str, color_map=None):
    names = list(drift_stats.keys())
    means = [v["mean"] for v in drift_stats.values()]
    stds = [v["std"] for v in drift_stats.values()]
    colors = [color_map.get(n, "gray") for n in names] if color_map else ["gray"] * len(names)
    global_drift = np.mean(means)
    means.append(global_drift)
    stds.append(np.std(means))
    names.append("AVERAGE")
    colors.append(color_map.get("Average", "black") if color_map else "black")

    plt.figure(figsize=(20, 8))
    # Higher bar = more drift (meaning lost faster)
    plt.bar(names, means, yerr=stds, capsize=5, color=colors)
    
    plt.grid(axis="y", alpha=0.3)
    plt.ylabel("ΔSP", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)
    plt.title("Average Semantic Drift per Transformation (ΔSP)", fontsize=24)
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "semantic_drift_rate_per_transform.png"))
    plt.close()

def drift_rate_per_persona(edges, embedder):
    results = []
    for e in edges:
        name = e.operator_metadata.get("persona_name", "unknown")
        stats = compute_semantic_drift_rate([e], embedder)
        results.append((name, stats["mean"]))
    
    # Sort by the mean value (index 1) in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def plot_drift_rate_persona(sorted_drift_data, vis_dir: str, color_map=None):
    """
    sorted_drift_data: List of tuples (name, mean_drift)
    """
    if not sorted_drift_data:
        print("[WARNING] No drift data to plot.")
        return

    # Unpack tuples
    names = [item[0] for item in sorted_drift_data]
    means = [item[1] for item in sorted_drift_data]
    
    n_instances = len(names)
    global_drift = np.mean(means)

    # Map colors
    if color_map:
        colors = [color_map.get(n, "gray") for n in names]
    else:
        colors = ["#e74c3c"] * n_instances  # A red-ish color for "Drift"

    plt.figure(figsize=(20, 8))
    
    # Plotting 21k bars: 
    # Use width=1.0 and linewidth=0 to prevent black border overlaps
    x_pos = np.arange(n_instances)
    plt.bar(x_pos, means, color=colors, width=1.0, linewidth=0, edgecolor='none')
    
    # Red dashed line for the average
    plt.axhline(global_drift, color='black', linestyle='--', linewidth=2, 
                label=f'AVERAGE ΔSP ({global_drift:.3f})')
    
    # Formatting
    plt.ylabel("Semantic Drift (ΔSP)", fontsize=22)
    plt.xlabel(f"Persona, sorted by drift", fontsize=20)
    plt.title("Semantic Drift Rate per Persona", fontsize=24)
    
    # CRITICAL: Hide x-ticks for 21,000 items
    plt.xticks([])
    plt.yticks(fontsize=18)
    
    plt.legend(fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "persona_level_semantic_drift.png"), dpi=300)
    plt.close()

# -- Best Delta Fitness per Iteration --
def best_delta_per_iteration(edges: List[MutationEdge]) -> Dict[int, float]:
    best = defaultdict(lambda: -np.inf)
    for e in edges:
        best[e.iteration_id] = max(best[e.iteration_id], e.delta_fitness)
    return dict(best)

def plot_best_delta(best_deltas: Dict[int, float], vis_dir: str):
    xs = sorted(best_deltas.keys())
    ys = [best_deltas[x] for x in xs]

    plt.figure(figsize=(16, 8))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Best Fitness Improvement", fontsize=16)
    plt.title("Best Attack Gain per Iteration", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "best_delta_per_iteration.png"))
    plt.close()

def save_persona_clusters(cluster_members, output_path):
    with open(output_path, "w") as f:
        json.dump(cluster_members, f, indent=2)

def save_cluster_summaries(cluster_summaries, output_path):
    with open(output_path, "w") as f:
        json.dump(cluster_summaries, f, indent=2)

def main_analysis(archive_path: List[str] | str, vis_dir: str, do_embeddings: bool = True):
    edges_raw = load_mutation_edges_from_archives(archive_path)
    embedder = EmbeddingCache()

    if do_embeddings:
        results = cluster_personas_hdbscan(
            edges_raw,
            embedder,
            field_weights=FIELD_WEIGHTS,
            vis_dir=vis_dir,
            min_cluster_size=40,
            min_samples=40,
        )

        print("[METRICS] Persona clustering completed.")

        persona_to_cluster = results["persona_to_cluster"]
        cluster_members = results["cluster_members"]
        cluster_summaries = results["cluster_summaries"]
        cluster_stability = results["cluster_stability"]

        save_persona_clusters(
            cluster_members,
            f"{vis_dir}/persona_clusters.json"
        )

        save_cluster_summaries(
            cluster_summaries,
            f"{vis_dir}/persona_cluster_summaries.json"
        )
        edges = relabel_edges_with_persona_clusters_hdbscan(edges_raw, persona_to_cluster)
        grouped = group_edges_by_operator(edges)
        print("[METRICS] Grouped edges by operator.")

        transform_names = list(grouped.keys())
        color_map = build_transform_color_map(transform_names)

        umap_results = generate_persona_umap(
            edges=edges_raw, 
            embedder=embedder, 
            cluster_summaries=cluster_summaries,
            weights=FIELD_WEIGHTS,
        )
        plot_embeddings(
            umap_results, 
            vis_dir,
        )

        return

        # ---- Edge-level ASR
        asr_per_transform = attack_success_rate_per_transform(grouped)
        plot_asr(asr_per_transform, vis_dir, color_map)
        print(f"[ANALYSIS] Completed ASR_G plotting.")

        # ---- Root-level ASR
        root_asr_per_transform = root_level_asr_per_transform(grouped)
        plot_root_level_asr(root_asr_per_transform, vis_dir, color_map)
        print(f"[ANALYSIS] Completed ASR_R plotting.")

        # ---- Semantic preservation
        preservation_per_transform = semantic_preservation_per_transform(grouped, embedder)
        plot_semantic_preservation(preservation_per_transform, vis_dir, color_map)
        # plot_semantic_preservation_violin(grouped, embedder, vis_dir, color_map)
        print(f"[ANALYSIS] Completed SP_G plotting.")

        # ---- Root-level semantic preservation
        root_preservation_per_transform = root_level_semantic_preservation_per_transform(grouped, embedder)
        plot_root_level_semantic_preservation(root_preservation_per_transform, vis_dir, color_map)
        print(f"[ANALYSIS] Completed SP_R plotting.")

        # ---- Heatmap
        plot_asr_preservation_heatmap(asr_per_transform, preservation_per_transform, vis_dir, type="edge")
        plot_asr_preservation_heatmap(root_asr_per_transform, root_preservation_per_transform, vis_dir, type="root")
        print(f"[ANALYSIS] Completed ASR vs SP heatmap plotting.")

        # ---- Scatter
        plot_asr_vs_preservation(asr_per_transform, preservation_per_transform, vis_dir, color_map, type="edge")
        plot_asr_vs_preservation(root_asr_per_transform, root_preservation_per_transform, vis_dir, color_map, type="root")
        print(f"[ANALYSIS] Completed ASR vs SP scatter plotting.")

        # ---- JCO
        jco_scores = jco_per_transform(grouped, embedder)
        plot_jco(jco_scores, vis_dir, color_map)
        print(f"[ANALYSIS] Completed JCO_G plotting.")

        # ---- Root-level JCO
        root_jco_scores = root_level_jco_per_transform(grouped, embedder)
        plot_root_level_jco(root_jco_scores, vis_dir, color_map)
        print(f"[ANALYSIS] Completed JCO_R plotting.")

        # ---- MEI
        mei_scores = mei_per_transform(grouped, embedder)
        plot_mei(mei_scores, vis_dir, color_map)
        print(f"[ANALYSIS] Completed MEI plotting.")

        # ---- Depth
        depth_stats = depth_of_success_per_transform(grouped)
        plot_depth_of_success(depth_stats, vis_dir, color_map)
        print(f"[ANALYSIS] Completed Depth of Success plotting.")

        # ---- Root-level Depth
        root_depth_stats = root_level_depth_of_success_per_transform(grouped)
        plot_root_level_depth_of_success(root_depth_stats, vis_dir, color_map)
        print(f"[ANALYSIS] Completed Root-Level Depth of Success plotting.")

        # ---- Metrics vs Depth
        depth_stats = cumulative_metrics_by_depth(edges, embedder)
        plot_cumulative_metrics_vs_depth(depth_stats, vis_dir)
        print(f"[ANALYSIS] Completed Metrics vs Depth plotting.")

        # ---- Drift rate
        drift_stats = drift_rate_per_transform(grouped, embedder)
        plot_drift_rate(drift_stats, vis_dir, color_map)
        print(f"[ANALYSIS] Completed Semantic Drift Rate plotting.")


        # collect metrics
        results = collect_results(
            edges=edges,
            grouped_edges=grouped,
            embedder=embedder,
        )

        # save
        save_results_json(
            results,
            output_path=f"{vis_dir}/persona_attack_metrics.json",
        )
    
    else:
        embedder = EmbeddingCache()

        # ---- Root-level semantic preservation
        root_preservation_per_persona = root_level_semantic_preservation_per_persona(edges_raw, embedder)
        plot_root_level_semantic_preservation_persona(root_preservation_per_persona, vis_dir)
        print(f"[ANALYSIS] Completed SP_R plotting.")


        # collect metrics
        results = collect_results(
            edges=edges_raw,
            grouped_edges=None,
            embedder=embedder,
        )

        # save
        save_results_json(
            results,
            output_path=f"{vis_dir}/persona_attack_metrics.json",
        )
        return

        edge_persona_depth = edge_level_depth_of_success_per_persona(edges_raw)
        plot_persona_edge_depth_of_success(edge_persona_depth, vis_dir)
        print(f"[ANALYSIS] Completed Edge-level Depth of Success plotting.")

        root_persona_depth_min = root_level_min_depth_of_success_per_persona(edges_raw)
        plot_root_level_min_depth_of_success(root_persona_depth_min, vis_dir)
        print(f"[ANALYSIS] Completed Root-level Min Depth of Success plotting.")

        root_persona_depth_avg = root_level_avg_depth_of_success_per_persona(edges_raw)
        plot_root_level_avg_depth_of_success(root_persona_depth_avg, vis_dir)
        print(f"[ANALYSIS] Completed Root-level Avg Depth of Success plotting.")

        # ---- Root-level semantic preservation
        root_preservation_per_persona = root_level_semantic_preservation_per_persona(edges_raw, embedder)
        plot_root_level_semantic_preservation_persona(root_preservation_per_persona, vis_dir)
        print(f"[ANALYSIS] Completed SP_R plotting.")

        # ---- Semantic preservation
        preservation_per_persona = semantic_preservation_per_persona(edges_raw, embedder)
        plot_semantic_preservation_persona(preservation_per_persona, vis_dir)
        print(f"[ANALYSIS] Completed SP_G plotting.")

        # ---- Root-level semantic preservation
        root_preservation_per_persona = root_level_semantic_preservation_per_persona(edges_raw, embedder)
        plot_root_level_semantic_preservation_persona(root_preservation_per_persona, vis_dir)
        print(f"[ANALYSIS] Completed SP_R plotting.")

        # ---- Metrics vs Depth
        depth_stats = cumulative_metrics_by_depth(edges_raw, embedder)
        plot_cumulative_metrics_vs_depth(depth_stats, vis_dir)
        print(f"[ANALYSIS] Completed Metrics vs Depth plotting.")

        # ---- Drift rate
        drift_stats = drift_rate_per_persona(edges_raw, embedder)
        plot_drift_rate_persona(drift_stats, vis_dir)
        print(f"[ANALYSIS] Completed Semantic Drift Rate plotting.")


        # collect metrics
        results = collect_results(
            edges=edges_raw,
            grouped_edges=None,
            embedder=embedder,
        )

        # save
        save_results_json(
            results,
            output_path=f"{vis_dir}/persona_attack_metrics.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Red-Teaming Archive Metrics")
    parser.add_argument(
        "--results_archive",
        type=str,
        default=None,
        help="Path to the red-teaming json archive"
    )
    parser.add_argument(
        "--results_folders", 
        type=str, 
        nargs='+',  # '+' means 1 or more arguments; '*' means 0 or more
        default=None, 
        help="List of folders separated by spaces"
    )
    args = parser.parse_args()

    archive_path = os.path.abspath(args.results_archive) if args.results_archive else None
    result_folders = args.results_folders

    do_embeddings = False

    if archive_path is not None:
        print(f"[ANALYSIS] Loading archive from: {archive_path}")

        # visualization directory
        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        main_analysis(archive_path, vis_dir, do_embeddings)
    else:
        paths = [os.path.abspath(os.path.join(folder, "red_teaming_archive.json")) for folder in result_folders]
        print(f"[ANALYSIS] Loading archives from: {paths}")
        vis_dir = "visualizations_combined"
        os.makedirs(vis_dir, exist_ok=True)
        
        main_analysis(paths, vis_dir, do_embeddings)