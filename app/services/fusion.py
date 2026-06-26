"""Fusion de listes de résultats (Reciprocal Rank Fusion).

La RRF combine plusieurs classements (ex. recherche dense + lexicale) sans
nécessiter de calibrer les échelles de score : seul le **rang** compte. Le score
fusionné d'un élément est la somme, sur chaque liste, de 1 / (k + rang).

Référence : Cormack et al., « Reciprocal Rank Fusion outperforms Condorcet… »
"""
from typing import Hashable, List, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]], rrf_k: int = 60
) -> List[Hashable]:
    """Fusionne plusieurs listes ordonnées en une seule (du plus pertinent au moins).

    Args:
        ranked_lists: listes d'identifiants, chacune triée par pertinence décroissante.
        rrf_k: constante d'atténuation (60 par défaut, valeur usuelle).

    Returns:
        La liste fusionnée des identifiants, sans doublon, triée par score RRF.
    """
    scores: dict = {}
    first_seen: dict = {}
    order = 0
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank + 1)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    # Tri par score décroissant ; en cas d'égalité, ordre de première apparition.
    return sorted(scores, key=lambda it: (-scores[it], first_seen[it]))
