"""Récupération lexicale (BM25) — une implémentation de l'interface Retriever.

BM25 classe les passages selon le recouvrement de mots-clés avec la requête.
Complémentaire de la recherche dense (sémantique) : il capte les correspondances
exactes de termes (noms propres, sigles, formules) que les embeddings ratent
parfois.
"""
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from services.interfaces import Retriever

import config

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    """Retriever lexical construit sur la liste de passages (chunks)."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks or []
        self._bm25 = None
        if self.chunks:
            corpus = [_tokenize(c["text"]) for c in self.chunks]
            self._bm25 = BM25Okapi(corpus)

    def list_sources(self) -> List[str]:
        seen: List[str] = []
        for c in self.chunks:
            if c["source"] not in seen:
                seen.append(c["source"])
        return seen

    def search(
        self, query: str, k: Optional[int] = None, sources: Optional[List[str]] = None
    ) -> List[Dict]:
        if self._bm25 is None:
            return []
        k = k or config.TOP_K
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        source_set = set(sources) if sources else None
        results: List[Dict] = []
        for i in order:
            chunk = self.chunks[i]
            if source_set is not None and chunk["source"] not in source_set:
                continue
            item = dict(chunk)
            item["score"] = float(scores[i])
            results.append(item)
            if len(results) >= k:
                break
        return results
