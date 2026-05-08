"""Neighbor stability: compare two retrievers on a fixed probe set.

This is the missing primitive RAGOps (Xu et al. 2025, arxiv:2506.03401)
calls out: "existing work provides limited support for observability in
the retrieval process of RAG applications." A pre/post snapshot of which
documents come back for a frozen set of probe queries lets you decide
whether an embedding model change, chunker change, or rerank change is
safe to deploy.

The function is pure: caller runs both retrievers, passes the result
dictionaries in. No vector-DB-specific code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityReport:
    """Per-set stability metrics over a probe set."""

    n_probes: int
    k: int
    mean_overlap_at_k: float
    """Mean ``|new ∩ old| / k`` across probes. 1.0 = identical top-k. 0.0 = disjoint."""

    mean_jaccard_at_k: float
    """Mean Jaccard similarity ``|new ∩ old| / |new ∪ old|`` across probes."""

    regression_probe_ids: tuple[str, ...]
    """Probe ids whose overlap fell below ``regression_threshold``."""

    @property
    def regression_count(self) -> int:
        return len(self.regression_probe_ids)

    def is_safe_to_deploy(
        self,
        *,
        min_mean_overlap: float = 0.85,
        max_regression_fraction: float = 0.05,
    ) -> bool:
        """Heuristic deploy gate.

        Returns True when both: mean overlap is at least ``min_mean_overlap``
        AND the fraction of regressed probes is at most ``max_regression_fraction``.
        """
        if self.n_probes == 0:
            return False
        regression_fraction = self.regression_count / self.n_probes
        return (
            self.mean_overlap_at_k >= min_mean_overlap
            and regression_fraction <= max_regression_fraction
        )


def neighbor_stability(
    old_results: dict[str, list[str]],
    new_results: dict[str, list[str]],
    *,
    k: int = 10,
    regression_threshold: float = 0.5,
) -> StabilityReport:
    """Compute :class:`StabilityReport` from two retrieval result sets.

    Both arguments map ``probe_id -> list of doc_id`` (top-k). Probe ids
    must agree across the two dicts; any extra ids in either are ignored.
    The first ``k`` doc ids of each list are compared.
    """
    common = set(old_results) & set(new_results)
    if not common:
        return StabilityReport(
            n_probes=0,
            k=k,
            mean_overlap_at_k=0.0,
            mean_jaccard_at_k=0.0,
            regression_probe_ids=(),
        )

    overlap_sum = 0.0
    jaccard_sum = 0.0
    regressions: list[str] = []
    for probe_id in sorted(common):
        old_topk = set(old_results[probe_id][:k])
        new_topk = set(new_results[probe_id][:k])
        intersect = len(old_topk & new_topk)
        union = len(old_topk | new_topk)
        overlap = intersect / k if k > 0 else 0.0
        jaccard = intersect / union if union > 0 else 0.0
        overlap_sum += overlap
        jaccard_sum += jaccard
        if overlap < regression_threshold:
            regressions.append(probe_id)

    n = len(common)
    return StabilityReport(
        n_probes=n,
        k=k,
        mean_overlap_at_k=overlap_sum / n,
        mean_jaccard_at_k=jaccard_sum / n,
        regression_probe_ids=tuple(regressions),
    )
