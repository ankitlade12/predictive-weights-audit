import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_loaders import NOAA_EDGE_PROVENANCE, load_noaa_edge_cards
from experiments.exp24_real_data_benchmark import collect_edge_cards
from ssm_causal.benchmark_metadata import (
    EdgeProvenance,
    build_adjacency_from_edges,
    edge_provenance_cards,
    leave_one_group_out,
)


def test_edge_provenance_cards_group_edges_by_label():
    edges = [
        EdgeProvenance("PNA", "ENSO", "causal", "Teleconnection literature"),
        EdgeProvenance("SOI", "ENSO", "definitional", "Same physical signal"),
        EdgeProvenance("XLRE", "XLF", "soft", "Finance commentary"),
    ]

    cards = edge_provenance_cards(edges)

    assert cards["causal"][0]["edge"] == "ENSO -> PNA"
    assert cards["definitional"][0]["source"] == "ENSO"
    assert cards["soft"][0]["rationale"] == "Finance commentary"


def test_build_adjacency_from_edges_can_filter_provenance_labels():
    names = ["ENSO", "SOI", "PNA"]
    edges = [
        EdgeProvenance("PNA", "ENSO", "causal", "Teleconnection literature"),
        EdgeProvenance("SOI", "ENSO", "definitional", "Same physical signal"),
    ]

    G = build_adjacency_from_edges(names, edges, include_labels={"causal"})

    assert G.shape == (3, 3)
    assert G[2, 0] == 1.0
    assert G[1, 0] == 0.0


def test_leave_one_group_out_reports_ranking_shifts():
    names = ["A", "B", "C"]
    edges = [
        EdgeProvenance("B", "A", "causal", "structural edge"),
        EdgeProvenance("C", "A", "proxy", "proxy edge"),
    ]
    scores = {
        "structural_method": np.array(
            [
                [0.0, 0.1, 0.1],
                [0.9, 0.0, 0.1],
                [0.7, 0.1, 0.0],
            ]
        ),
        "proxy_method": np.array(
            [
                [0.0, 0.1, 0.1],
                [0.2, 0.0, 0.1],
                [0.8, 0.1, 0.0],
            ]
        ),
    }

    result = leave_one_group_out(names, edges, scores, group_by="label")

    assert result["baseline"]["ranking"][0]["method"] == "structural_method"
    proxy_removed = result["leave_out"]["proxy"]
    assert proxy_removed["remaining_edges"] == 1
    assert proxy_removed["ranking"][0]["method"] == "structural_method"
    causal_removed = result["leave_out"]["causal"]
    assert causal_removed["ranking"][0]["method"] == "proxy_method"


def test_noaa_loader_exposes_definitional_edge_cards():
    cards = load_noaa_edge_cards(include_definitional=True)

    assert any(edge.label == "definitional" for edge in NOAA_EDGE_PROVENANCE)
    assert cards["definitional"][0]["edge"] in {"ENSO -> SOI", "SOI -> ENSO"}


def test_real_benchmark_collects_edge_cards_for_soft_ground_truth():
    cards = collect_edge_cards(["climate", "finance", "lorenz96"])

    assert "climate" in cards
    assert "finance" in cards
    assert "lorenz96" not in cards
    assert "causal" in cards["climate"]
    assert "soft" in cards["finance"]
