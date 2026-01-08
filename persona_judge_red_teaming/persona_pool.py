from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib
import random
import numpy as np


# ----------------
# Helper functions
# ----------------

# Normalize whitespace and casing for stable signatures
def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


# Stable signature for deduplication/identity
def _stable_sig(text: str) -> str:
    # note: sha256 is deterministic and collision-resistant for this use
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


# Safe, best-effort JSON parsing with extraction fallback
def _safe_parse_json_dict(raw_text: str) -> Tuple[Dict[str, Any], str]:
    raw = (raw_text or "").strip()

    # Try strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "json_ok"
    except Exception:
        pass

    # Try extracting the outermost {...}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj, "json_extracted"
    except Exception:
        pass

    # Fallback: unstructured persona
    return {
        "name": "unstructured_persona",
        "style": raw,
        "tone": "unknown",
        "domain": "unknown",
        "obfuscation_strategy": "unknown",
    }, "raw_text"


# Canonical string used for embedding + identity
def _persona_to_canonical_text(persona_fields: Dict[str, Any], parse_status: str) -> str:
    # If parsing failed, preserve the raw description in "style"
    if parse_status == "raw_text":
        return str(persona_fields.get("style", "")).strip()

    keys = ["name", "style", "tone", "domain", "obfuscation_strategy"]
    parts: List[str] = []
    for k in keys:
        v = persona_fields.get(k, "")
        if v is None:
            continue
        v_str = str(v).strip()
        if v_str:
            parts.append(f"{k}: {v_str}")
    return " | ".join(parts).strip()


# Robust embedding normalization (works whether embedder normalizes or not)
def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return vec / denom


# Cosine similarity for normalized vectors (fast, stable)
def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    # Assumes both are normalized; if not, still works but slightly slower/less stable.
    return float(np.dot(u, v))


# ----------------------
# Persona representation
# ----------------------

@dataclass
class Persona:
    # A stable persona id derived from canonical text
    pid: str

    # Canonical text representation used for embedding and similarity
    canonical_text: str

    # Raw LLM output and parsing status are kept for traceability
    raw_output: str
    parse_status: str

    # Structured fields used by your attacker utils (json.dumps(persona_dict))
    fields: Dict[str, Any]

    # Embedding is stored internally; do not serialize into archives
    embedding: np.ndarray = field(repr=False)

    # Lightweight performance stats for selection
    uses: int = 0
    successes: int = 0
    mean_delta: float = 0.0
    attempts: int = 0
    success_depth_sum: float = 0.0
    success_count: int = 0


    # Optional: last update iteration (useful for aging/eviction)
    last_updated_iter: int = -1

    # ------------
    # Constructors
    # ------------

    @staticmethod
    def from_llm_output(
        raw_output: str,
        embedder: Any,
        *,
        iteration_idx: int = -1,
    ) -> "Persona":
        # Parse persona (robust)
        fields, status = _safe_parse_json_dict(raw_output)

        # Canonical string representation
        canonical = _persona_to_canonical_text(fields, status)

        # Stable id
        pid = _stable_sig(canonical)

        # Embed canonical text
        # Note: embedder should be an object with .encode(List[str], normalize_embeddings=...)
        emb = embedder.encode([canonical], normalize_embeddings=True)[0]
        emb = np.asarray(emb, dtype=np.float32)
        emb = _l2_normalize(emb)

        return Persona(
            pid=pid,
            canonical_text=canonical,
            raw_output=(raw_output or "").strip(),
            parse_status=status,
            fields=fields,
            embedding=emb,
            uses=0,
            successes=0,
            mean_delta=0.0,
            last_updated_iter=iteration_idx,
        )

    @staticmethod
    def from_fields(
        fields: Dict[str, Any],
        embedder: Any,
        *,
        raw_output: str = "",
        parse_status: str = "manual",
        iteration_idx: int = -1,
    ) -> "Persona":
        canonical = _persona_to_canonical_text(fields, parse_status if parse_status else "manual")
        pid = _stable_sig(canonical)
        emb = embedder.encode([canonical], normalize_embeddings=True)[0]
        emb = np.asarray(emb, dtype=np.float32)
        emb = _l2_normalize(emb)
        return Persona(
            pid=pid,
            canonical_text=canonical,
            raw_output=(raw_output or "").strip(),
            parse_status=parse_status if parse_status else "manual",
            fields=dict(fields),
            embedding=emb,
            uses=0,
            successes=0,
            mean_delta=0.0,
            last_updated_iter=iteration_idx,
        )

    # -----------------------
    # Stats and serialization
    # -----------------------

    def update_stats(self, *, delta: float, success: bool, depth: int, iteration_idx: int) -> None:
        # Update usage counters
        self.uses += 1
        self.attempts += 1
        if success:
            self.successes += 1
            self.success_count += 1
            self.success_depth_sum += float(depth)

        # Update running mean delta
        # Note: delta is expected to be (parent_score - child_score) or (item.score - refined_score), so larger is better
        self.mean_delta = ((self.mean_delta * (self.uses - 1) + float(delta)) / float(self.uses))

        # Update last updated iteration
        self.last_updated_iter = iteration_idx
    
    def fitness(self, *, alpha: float = 2.0, min_attempts: int = 3) -> float:
        """
        Composite evolutionary fitness.
        """
        if self.attempts < min_attempts:
            return 0.0  # prevent early lucky personas from dominating

        success_rate = self.successes / max(self.attempts, 1)

        if self.successes > 0:
            avg_depth = self.success_depth_sum / self.successes
        else:
            avg_depth = float("inf")

        # efficiency = 1.0 / (1.0 + avg_depth)

        return float(self.mean_delta) * (success_rate ** alpha) # * efficiency

    def utility(self) -> float:
        # Simple utility: mean score improvement; stable and directly tied to your logged delta_fitness
        return float(self.mean_delta)

    def as_dict_for_attacker(self) -> Dict[str, Any]:
        # This is what you pass to utils.get_persona_mutation_messages and json.dumps
        # Keep only the intended fields (do not include embeddings/stats)
        return dict(self.fields)

    def as_dict_for_archive(self) -> Dict[str, Any]:
        # Safe to store in your JSON archives: no numpy arrays, no non-serializable objects
        # Includes pid and minimal metadata for later analysis
        d = dict(self.fields)
        d["_pid"] = self.pid
        d["_canonical_text"] = self.canonical_text
        d["_parse_status"] = self.parse_status
        d["_uses"] = self.uses
        d["_successes"] = self.successes
        d["_mean_delta"] = self.mean_delta
        return d

class PersonaPool:

    def __init__(
        self,
        *,
        embedder: Any,
        max_active: int = 30,
        merge_threshold: float = 0.90,
        rng: Optional[random.Random] = None,
    ):
        # Embedder should be a SentenceTransformer (or compatible) already initialized in main and passed in
        self.embedder = embedder
        self.max_active = int(max_active)
        self.merge_threshold = float(merge_threshold)
        self.rng = rng if rng is not None else random.Random()

        # Active and history store Persona objects
        self.active: List[Persona] = []
        self.history: List[Persona] = []

        # Quick lookup by pid
        self._by_pid: Dict[str, Persona] = {}

        # Minimum attempts before fitness is considered
        self.min_attempts = 3

    # --------------------------
    # Core similarity operations
    # --------------------------

    def similarity(self, a: Persona, b: Persona) -> float:
        return _cosine_sim(a.embedding, b.embedding)

    def _closest_active(self, persona: Persona) -> Tuple[Optional[Persona], float]:
        if not self.active:
            return None, -1.0

        best_p: Optional[Persona] = None
        best_sim = -1.0
        for p in self.active:
            sim = _cosine_sim(persona.embedding, p.embedding)
            if sim > best_sim:
                best_sim = sim
                best_p = p
        return best_p, float(best_sim)

    # -------
    # Seeding
    # -------

    def seed(self, seed_personas: List[Dict[str, Any]], *, iteration_idx: int = -1) -> None:
        for fields in seed_personas:
            persona = Persona.from_fields(fields, self.embedder, parse_status="seed", iteration_idx=iteration_idx)
            persona.attempts = 1  # avoid min_attempts blocking
            persona.successes = 1
            persona.mean_delta = 0.01  # small positive to favor seeding
            # treat seed personas as admitted without performance stats
            self._admit_new(persona)

    # ------------------------------
    # Admission / merging / eviction
    # ------------------------------

    def consider_candidate_from_llm(
        self,
        raw_output: str,
        *,
        delta: float,
        success: bool,
        depth: int,
        iteration_idx: int,
    ) -> Tuple[Persona, str, float]:
        """
        Parse+embed a persona candidate and decide:
        - "merged": merged into an existing similar persona (possibly replaced if better)
        - "admitted": added to active pool (possibly via eviction)
        - "discarded": not admitted
        Returns: (persona_used, action, similarity_to_closest)
        """
        candidate = Persona.from_llm_output(raw_output, self.embedder, iteration_idx=iteration_idx)
        candidate.update_stats(delta=delta, success=success, depth=depth, iteration_idx=iteration_idx)

        closest, sim = self._closest_active(candidate)

        # Merge path
        if closest is not None and sim >= self.merge_threshold:
            # keep the better one by fitness; update stats on the kept persona
            if candidate.fitness() > closest.fitness():
                # Replace closest with candidate
                self._replace_active(old=closest, new=candidate)
                return candidate, "merged_replaced", float(sim)
            else:
                # Merge: keep closest, but still record candidate in history for auditability
                self.history.append(candidate)
                return closest, "merged_kept", float(sim)

        # Novel persona path: performance-gated admission
        action = self._maybe_admit(candidate)
        return candidate if action != "discarded" else (closest if closest is not None else candidate), action, float(sim)

    def _maybe_admit(self, candidate: Persona) -> str:
        # If pool has room, admit
        if len(self.active) < self.max_active:
            self._admit_new(candidate)
            return "admitted"

        # Pool full: evict worst if candidate is better
        # worst = min(self.active, key=lambda p: p.fitness())
        worst = min(
            self.active,
            key=lambda p: (p.attempts < self.min_attempts, p.fitness())
        )
        if candidate.fitness() > worst.fitness():
            self._evict(worst)
            self._admit_new(candidate)
            return "admitted_evicted"

        # Otherwise discard (but keep in history for traceability)
        self.history.append(candidate)
        return "discarded"

    def _admit_new(self, persona: Persona) -> None:
        # Deduplicate by pid
        if persona.pid in self._by_pid:
            # If already known, do not duplicate; but optionally update last_updated
            existing = self._by_pid[persona.pid]
            existing.last_updated_iter = max(existing.last_updated_iter, persona.last_updated_iter)
            return

        self.active.append(persona)
        self.history.append(persona)
        self._by_pid[persona.pid] = persona

        # Enforce max_active just in case caller admitted too many
        if len(self.active) > self.max_active:
            worst = min(self.active, key=lambda p: p.fitness())
            self._evict(worst)

    def _replace_active(self, *, old: Persona, new: Persona) -> None:
        # Remove old
        self._evict(old)
        # Add new
        self._admit_new(new)

    def _evict(self, persona: Persona) -> None:
        # Remove from active only; keep in history
        self.active = [p for p in self.active if p.pid != persona.pid]
        # Keep _by_pid mapping for archival continuity; do not delete

    # --------
    # Sampling
    # --------

    def sample_active(self, *, mode: str = "weighted") -> Persona:
        if not self.active:
            raise RuntimeError("PersonaPool has no active personas.")

        if mode == "uniform":
            return self.rng.choice(self.active)

        # fitness-proportional sampling with temperature
        fitnesses = np.array([p.fitness() for p in self.active], dtype=np.float64)

        if np.all(fitnesses <= 0):
            return self.rng.choice(self.active)

        temperature = 0.5  # lower = stronger selection
        scaled = fitnesses / temperature
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()

        return self.rng.choices(self.active, weights=probs, k=1)[0]
    
    # -----------------
    # Utility accessors
    # -----------------

    def active_size(self) -> int:
        return len(self.active)

    def history_size(self) -> int:
        return len(self.history)

    def top_active(self, k: int = 5) -> List[Persona]:
        return sorted(self.active, key=lambda p: p.fitness(), reverse=True)[: int(k)]

    # ----------------------------------
    # Export helpers (for JSON archives)
    # ----------------------------------

    def persona_for_attacker(self, persona: Persona) -> Dict[str, Any]:
        return persona.as_dict_for_attacker()

    def persona_for_archive(self, persona: Persona) -> Dict[str, Any]:
        return persona.as_dict_for_archive()
