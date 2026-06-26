"""Tests unitaires des métriques d'évaluation (hors-ligne, déterministes)."""
from metrics import (
    aggregate,
    evaluate_query,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k():
    assert hit_at_k(["a.txt", "b.txt", "c.txt"], ["b.txt"], k=3) == 1.0
    assert hit_at_k(["a.txt", "b.txt"], ["z.txt"], k=2) == 0.0
    # Hors fenêtre top-k
    assert hit_at_k(["a.txt", "b.txt", "c.txt"], ["c.txt"], k=2) == 0.0


def test_recall_at_k():
    # 1 pertinent sur 2 attendus dans le top-2
    assert recall_at_k(["a.txt", "x.txt"], ["a.txt", "b.txt"], k=2) == 0.5
    assert recall_at_k(["a.txt", "b.txt"], ["a.txt", "b.txt"], k=2) == 1.0
    assert recall_at_k(["x.txt"], [], k=1) == 0.0


def test_precision_at_k():
    # 1 pertinent parmi 2 récupérés
    assert precision_at_k(["a.txt", "x.txt"], ["a.txt"], k=2) == 0.5
    # top-k plus grand que la liste : on divise par la taille réelle
    assert precision_at_k(["a.txt"], ["a.txt"], k=5) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x.txt", "a.txt"], ["a.txt"]) == 0.5  # 1/2
    assert reciprocal_rank(["a.txt", "x.txt"], ["a.txt"]) == 1.0  # 1/1
    assert reciprocal_rank(["x.txt", "y.txt"], ["a.txt"]) == 0.0


def test_evaluate_query_structure():
    res = evaluate_query(["a.txt", "b.txt"], ["a.txt"], k=2)
    assert set(res) == {"hit@k", "recall@k", "precision@k", "mrr"}
    assert res["hit@k"] == 1.0
    assert res["mrr"] == 1.0


def test_aggregate_moyenne():
    q1 = {"hit@k": 1.0, "recall@k": 1.0, "precision@k": 0.5, "mrr": 1.0}
    q2 = {"hit@k": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0}
    agg = aggregate([q1, q2])
    assert agg["hit@k"] == 0.5
    assert agg["precision@k"] == 0.25


def test_aggregate_vide():
    agg = aggregate([])
    assert agg["hit@k"] == 0.0
