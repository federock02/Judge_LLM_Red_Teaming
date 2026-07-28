"""
test_asr_star.py — dependency-light checks for the ASR* metric functions.

metrics.py imports sentence_transformers at module scope, which is only available
inside the cluster venv. The ASR* functions themselves never touch the model: they
only call embedder.cosine(a, b). So we stub out sentence_transformers before the
import and drive the metrics with a fake embedder whose cosines we control.

This verifies the gating logic, the root/edge distinction, the depth accumulation
and the monotonicity of the threshold sweep, all without a GPU or a download.

Run from the directory containing metrics.py:
    python test_asr_star.py
"""
import sys
import types


# --- stub sentence_transformers so `import metrics` works without the venv -----
_stub = types.ModuleType("sentence_transformers")
class _FakeST:                                    # noqa: D401 - test stub
    def __init__(self, *a, **k):
        raise RuntimeError("SentenceTransformer must not be constructed in tests")
_stub.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", _stub)

import metrics as M                                # noqa: E402


class FakeEmbedder:
    """Returns a caller-supplied cosine for each (a, b) text pair."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def cosine(self, a, b):
        self.calls += 1
        return self.table[(a, b)]


def edge(root, child, *, success, depth=1, parent=None, op="op"):
    return M.MutationEdge(
        iteration_id=0,
        refinement_iter=depth,
        root_prompt=root,
        root_fitness=0.9,
        source_dataset="test",
        parent_prompt=parent if parent is not None else root,
        parent_fitness=0.9,
        child_prompt=child,
        child_fitness=0.1 if success else 0.9,
        delta_fitness=0.8 if success else 0.0,
        success=success,
        operator_type="transformation",
        operator_name=op,
        operator_metadata={},
        is_root_edge=(parent is None),
    )


def check(name, got, want):
    ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        raise AssertionError(name)


def main():
    # Root R1: one bypass with high SP (valid), one with low SP (evasion by degradation)
    # Root R2: one bypass, but low SP only              -> counts for bypass, not ASR*
    # Root R3: no bypass at all
    edges = [
        edge("R1", "c1", success=True,  depth=1),   # cos .80 -> valid
        edge("R1", "c2", success=True,  depth=2),   # cos .20 -> gated out
        edge("R1", "c3", success=False, depth=1),   # cos .90 but no evasion
        edge("R2", "c4", success=True,  depth=3),   # cos .30 -> gated out
        edge("R2", "c5", success=False, depth=1),
        edge("R3", "c6", success=False, depth=1),
    ]
    cos = {
        ("R1", "c1"): 0.80, ("R1", "c2"): 0.20, ("R1", "c3"): 0.90,
        ("R2", "c4"): 0.30, ("R2", "c5"): 0.55,
        ("R3", "c6"): 0.95,
    }
    emb = FakeEmbedder(cos)

    print("\n[1] gating semantics")
    # 3 of 6 edges evade; only 1 of those also preserves semantics.
    check("edge-level bypass rate", M.attack_success_rate(edges), 3 / 6)
    check("edge-level ASR*", M.attack_success_rate_star(edges, emb, 0.5), 1 / 6)
    # All 3 roots have >=1 bypass... no: R3 has none. R1, R2 do.
    check("root-level bypass rate", M.root_level_attack_success_rate(edges), 2 / 3)
    # Only R1 has a semantics-preserving bypass.
    check("root-level ASR*", M.root_level_attack_success_rate_star(edges, emb, 0.5), 1 / 3)

    print("\n[2] a high-SP edge that does not evade must not count")
    # R3's only edge has cos .95 but success=False.
    r3 = [e for e in edges if e.root_prompt == "R3"]
    check("R3 ASR*", M.root_level_attack_success_rate_star(r3, emb, 0.5), 0.0)

    print("\n[3] threshold extremes bracket the ungated metric")
    check("tau=0 reproduces bypass rate (edge)",
          M.attack_success_rate_star(edges, emb, 0.0), M.attack_success_rate(edges))
    check("tau=0 reproduces bypass rate (root)",
          M.root_level_attack_success_rate_star(edges, emb, 0.0),
          M.root_level_attack_success_rate(edges))
    check("tau>1 gives zero", M.root_level_attack_success_rate_star(edges, emb, 1.01), 0.0)

    print("\n[4] ASR* is monotone non-increasing in tau")
    taus = (0.1, 0.25, 0.5, 0.85)
    for name, fn in (("edge", M.attack_success_rate_star),
                     ("root", M.root_level_attack_success_rate_star)):
        vals = [fn(edges, emb, t) for t in taus]
        print(f"    {name}-level: {vals}")
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), \
            f"{name} not monotone: {vals}"
    print("  PASS  monotonicity")

    print("\n[5] cumulative ASR* by depth")
    # R1's valid success is at depth 1, so ASR*@d = 1/3 for every d >= 1.
    cum = M.cumulative_root_asr_star_by_depth(edges, emb, 0.5)
    check("ASR*@1", cum[1], 1 / 3)
    check("ASR*@5", cum[5], 1 / 3)
    # Ungated: R1 at depth 1, R2 at depth 3.
    cum_b = M.cumulative_root_asr_by_depth(edges)
    check("bypass@1", cum_b[1], 1 / 3)
    check("bypass@3", cum_b[3], 2 / 3)
    print("  (ASR* <= bypass at every depth)")
    assert all(cum[d] <= cum_b[d] + 1e-12 for d in cum), "ASR* exceeded bypass rate"
    print("  PASS  ASR* dominated by bypass rate at all depths")

    print("\n[6] per-operator variants")
    grouped = M.group_edges_by_operator(edges)
    check("operator count", len(grouped), 1)
    check("per-op root ASR*",
          M.root_level_asr_star_per_operator(grouped, emb, 0.5)["op"], 1 / 3)

    print("\n[7] empty input is handled")
    check("empty edge ASR*", M.attack_success_rate_star([], emb, 0.5), 0.0)
    check("empty root ASR*", M.root_level_attack_success_rate_star([], emb, 0.5), 0.0)
    check("empty cumulative", M.cumulative_root_asr_star_by_depth([], emb, 0.5), {})

    print(f"\nAll ASR* checks passed ({emb.calls} cosine lookups).\n")


if __name__ == "__main__":
    main()
