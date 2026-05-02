"""Structured metadata helpers for benchmark ground-truth edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .utils import compute_metrics

ProvenanceLabel = Literal["causal", "definitional", "proxy", "soft"]


@dataclass(frozen=True)
class EdgeProvenance:
    """Ground-truth edge with provenance needed for sensitivity audits."""

    target: str
    source: str
    label: ProvenanceLabel
    rationale: str
    reference: str | None = None
    group: str | None = None

    @property
    def edge(self) -> str:
        return f"{self.source} -> {self.target}"

    @property
    def audit_group(self) -> str:
        return self.group or self.label


def edge_provenance_cards(edges: Iterable[EdgeProvenance]) -> dict[str, list[dict[str, str]]]:
    """Group edge provenance records into JSON-serializable cards."""
    cards: dict[str, list[dict[str, str]]] = {}
    for edge in edges:
        card = {
            "edge": edge.edge,
            "target": edge.target,
            "source": edge.source,
            "label": edge.label,
            "group": edge.audit_group,
            "rationale": edge.rationale,
        }
        if edge.reference:
            card["reference"] = edge.reference
        cards.setdefault(edge.label, []).append(card)
    return cards


def build_adjacency_from_edges(
    names: list[str],
    edges: Iterable[EdgeProvenance],
    include_labels: set[str] | None = None,
) -> np.ndarray:
    """Build G[target, source] from edge metadata."""
    name_idx = {name: idx for idx, name in enumerate(names)}
    G = np.zeros((len(names), len(names)), dtype=np.float32)

    for edge in edges:
        if include_labels is not None and edge.label not in include_labels:
            continue
        if edge.target not in name_idx or edge.source not in name_idx:
            continue
        G[name_idx[edge.target], name_idx[edge.source]] = 1.0

    np.fill_diagonal(G, 0.0)
    return G


def _rank_methods(scores: dict[str, np.ndarray], G: np.ndarray) -> list[dict[str, float | str]]:
    ranking = []
    for method, S in scores.items():
        metrics = compute_metrics(S, G)
        auroc = float(metrics["auroc"])
        ranking.append({"method": method, "auroc": auroc})
    ranking.sort(
        key=lambda row: (np.nan_to_num(row["auroc"], nan=-1.0), row["method"]),
        reverse=True,
    )
    return ranking


def leave_one_group_out(
    names: list[str],
    edges: list[EdgeProvenance],
    scores: dict[str, np.ndarray],
    group_by: Literal["label", "group"] = "group",
) -> dict[str, object]:
    """
    Recompute method rankings after removing each edge provenance group.

    This is intended for soft-ground-truth datasets where a small set of
    definitional/proxy edges can dominate AUROC.
    """
    baseline_G = build_adjacency_from_edges(names, edges)
    groups = sorted({edge.label if group_by == "label" else edge.audit_group for edge in edges})

    result: dict[str, object] = {
        "baseline": {
            "included_edges": int(baseline_G.sum()),
            "ranking": _rank_methods(scores, baseline_G),
        },
        "leave_out": {},
    }

    leave_out: dict[str, object] = {}
    for group in groups:
        kept_edges = [
            edge
            for edge in edges
            if (edge.label if group_by == "label" else edge.audit_group) != group
        ]
        G = build_adjacency_from_edges(names, kept_edges)
        leave_out[group] = {
            "remaining_edges": int(G.sum()),
            "ranking": _rank_methods(scores, G),
        }
    result["leave_out"] = leave_out
    return result
