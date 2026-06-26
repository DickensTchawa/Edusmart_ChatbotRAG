"""Métriques d'évaluation de la récupération (retrieval) d'un système RAG.

Toutes les fonctions sont pures et déterministes : elles ne dépendent que des
listes de sources récupérées et attendues. Elles sont donc testables hors-ligne,
sans modèle ni réseau.

Convention : `retrieved` est la liste ORDONNÉE des documents récupérés (du plus
pertinent au moins pertinent, sans doublon) ; `expected` est l'ensemble des
documents réellement pertinents pour la question.
"""
from typing import Dict, List, Sequence


def _topk(retrieved: Sequence[str], k: int) -> List[str]:
    return list(retrieved[:k])


def hit_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """1.0 si au moins un document pertinent figure dans le top-k, sinon 0.0."""
    topk = set(_topk(retrieved, k))
    return 1.0 if topk & set(expected) else 0.0


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Proportion des documents pertinents retrouvés dans le top-k."""
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    found = set(_topk(retrieved, k)) & expected_set
    return len(found) / len(expected_set)


def precision_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Proportion de documents pertinents parmi les k premiers récupérés."""
    topk = _topk(retrieved, k)
    if not topk:
        return 0.0
    expected_set = set(expected)
    relevant = sum(1 for s in topk if s in expected_set)
    return relevant / len(topk)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Inverse du rang du premier document pertinent (0.0 si aucun)."""
    expected_set = set(expected)
    for i, s in enumerate(retrieved, start=1):
        if s in expected_set:
            return 1.0 / i
    return 0.0


def evaluate_query(retrieved: Sequence[str], expected: Sequence[str], k: int) -> Dict[str, float]:
    """Calcule toutes les métriques pour une seule question."""
    return {
        "hit@k": hit_at_k(retrieved, expected, k),
        "recall@k": recall_at_k(retrieved, expected, k),
        "precision@k": precision_at_k(retrieved, expected, k),
        "mrr": reciprocal_rank(retrieved, expected),
    }


def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Moyenne chaque métrique sur l'ensemble des questions."""
    if not per_query:
        return {"hit@k": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
