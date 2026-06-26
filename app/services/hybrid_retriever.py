"""Récupération hybride : fusion d'un retriever dense et d'un retriever lexical.

HybridRetriever implémente l'interface Retriever en combinant deux autres
Retriever (patron *Composite* / *Strategy*) :
1. chaque sous-retriever propose ses meilleurs candidats ;
2. les deux classements sont fusionnés par Reciprocal Rank Fusion ;
3. un *reranker* optionnel peut réordonner finement les candidats.

Comme il respecte l'interface Retriever, il est interchangeable avec le retriever
dense sans modifier le contrôleur.
"""
from typing import Callable, Dict, List, Optional

from services.fusion import reciprocal_rank_fusion
from services.interfaces import Retriever

import config

# Un reranker est un appelable (query, passages, k) -> passages réordonnés.
Reranker = Callable[[str, List[Dict], int], List[Dict]]


def _key(passage: Dict):
    return (passage.get("source"), passage.get("chunk_id"))


class HybridRetriever(Retriever):
    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        reranker: Optional[Reranker] = None,
        candidate_k: Optional[int] = None,
    ):
        self.dense = dense
        self.lexical = lexical
        self.reranker = reranker
        self.candidate_k = candidate_k or config.HYBRID_CANDIDATES

    def list_sources(self) -> List[str]:
        return self.dense.list_sources()

    def search(
        self, query: str, k: Optional[int] = None, sources: Optional[List[str]] = None
    ) -> List[Dict]:
        k = k or config.TOP_K
        dense_res = self.dense.search(query, k=self.candidate_k, sources=sources)
        lexical_res = self.lexical.search(query, k=self.candidate_k, sources=sources)

        # Index des passages par identifiant (première occurrence conservée).
        by_id: Dict = {}
        for p in (*dense_res, *lexical_res):
            by_id.setdefault(_key(p), p)

        fused_ids = reciprocal_rank_fusion(
            [[_key(p) for p in dense_res], [_key(p) for p in lexical_res]]
        )
        passages = [by_id[i] for i in fused_ids]

        if self.reranker is not None:
            passages = self.reranker(query, passages, k)

        return passages[:k]
