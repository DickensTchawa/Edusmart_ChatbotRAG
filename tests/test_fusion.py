"""Tests de la fusion Reciprocal Rank Fusion (hors-ligne)."""
from services.fusion import reciprocal_rank_fusion


def test_rrf_favorise_les_elements_communs():
    # 'b' apparaît dans les deux listes -> meilleur score fusionné.
    dense = ["a", "b", "c"]
    lexical = ["b", "d"]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}  # union, sans doublon


def test_rrf_liste_unique_preserve_l_ordre():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_vide():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
