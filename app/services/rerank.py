"""Re-ranking optionnel par cross-encoder.

Un *cross-encoder* évalue conjointement (question, passage) et fournit un score
de pertinence bien plus fin qu'une similarité de vecteurs indépendants. Il est
appliqué après la fusion, sur un petit nombre de candidats (coûteux mais précis).

Désactivé par défaut (téléchargement d'un modèle). Le modèle est chargé
paresseusement au premier appel. L'objet est un simple appelable
(query, passages, k) -> passages réordonnés, compatible avec HybridRetriever.
"""
from typing import Dict, List

import config


class CrossEncoderReranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"[rerank] Chargement du cross-encoder : {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def __call__(self, query: str, passages: List[Dict], k: int) -> List[Dict]:
        if not passages:
            return passages
        pairs = [(query, p["text"]) for p in passages]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(passages, scores), key=lambda ps: float(ps[1]), reverse=True
        )
        out = []
        for passage, score in ranked[:k]:
            item = dict(passage)
            item["rerank_score"] = float(score)
            out.append(item)
        return out
