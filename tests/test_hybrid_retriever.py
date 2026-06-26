"""Tests du retriever hybride (fusion dense + lexical), entièrement hors-ligne.

Les deux sous-retrievers sont des doublures déterministes : on valide la fusion,
la déduplication, le re-ranking optionnel et la délégation de list_sources.
"""
from typing import Dict, List, Optional

from services.hybrid_retriever import HybridRetriever
from services.interfaces import Retriever


class StubRetriever(Retriever):
    def __init__(self, passages: List[Dict]):
        self._passages = passages

    def search(self, query, k=None, sources=None) -> List[Dict]:
        return self._passages[: (k or len(self._passages))]

    def list_sources(self) -> List[str]:
        return sorted({p["source"] for p in self._passages})


def _p(source):
    return {"source": source, "chunk_id": 0, "text": source.upper()}


def test_hybride_fusionne_et_priorise_les_communs():
    dense = StubRetriever([_p("a.txt"), _p("b.txt"), _p("c.txt")])
    lexical = StubRetriever([_p("b.txt"), _p("d.txt")])
    hybrid = HybridRetriever(dense, lexical)
    res = hybrid.search("q", k=2)
    # b.txt présent dans les deux listes -> premier ; pas de doublon.
    assert res[0]["source"] == "b.txt"
    assert len(res) == 2
    assert len({p["source"] for p in res}) == 2


def test_hybride_applique_le_reranker():
    dense = StubRetriever([_p("a.txt"), _p("b.txt")])
    lexical = StubRetriever([_p("b.txt")])

    def reverse_reranker(query: str, passages: List[Dict], k: int) -> List[Dict]:
        return list(reversed(passages))[:k]

    hybrid = HybridRetriever(dense, lexical, reranker=reverse_reranker)
    res = hybrid.search("q", k=2)
    assert [p["source"] for p in res] == list(reversed([p["source"] for p in
            HybridRetriever(dense, lexical).search("q", k=99)]))[:2]


def test_hybride_delegue_list_sources():
    dense = StubRetriever([_p("a.txt"), _p("b.txt")])
    lexical = StubRetriever([_p("c.txt")])
    hybrid = HybridRetriever(dense, lexical)
    assert hybrid.list_sources() == ["a.txt", "b.txt"]


def test_hybride_est_un_retriever():
    hybrid = HybridRetriever(StubRetriever([]), StubRetriever([]))
    assert isinstance(hybrid, Retriever)
