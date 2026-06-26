"""Contrats (interfaces) des composants interchangeables du chatbot.

Programmer par interfaces illustre le **principe d'inversion des dépendances**
(le « D » de SOLID) : le contrôleur dépend de ces abstractions, pas des
implémentations concrètes. On peut ainsi remplacer le moteur de recherche
(FAISS → Chroma, pgvector…) ou le fournisseur de LLM sans modifier le reste
(patrons *Strategy* et *Adapter*).
"""
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional


class Retriever(ABC):
    """Composant de récupération de passages pertinents (le « R » de RAG)."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Retourne les passages les plus pertinents pour `query`.

        `sources`, s'il est fourni, restreint la recherche à ces documents.
        Chaque passage est un dict contenant au moins `text`, `source` et `score`.
        """
        raise NotImplementedError

    @abstractmethod
    def list_sources(self) -> List[str]:
        """Retourne la liste distincte des documents indexés."""
        raise NotImplementedError


class LLMProvider(ABC):
    """Composant de génération de la réponse (le « G » de RAG)."""

    @abstractmethod
    def answer(
        self, question: str, passages: List[Dict], history: Optional[List[Dict]] = None
    ) -> str:
        """Génère une réponse en langage naturel à partir des passages fournis.

        `history`, s'il est fourni, contient les tours précédents de la
        conversation (`{role, content}`) pour gérer les questions de suivi.
        """
        raise NotImplementedError

    def answer_stream(
        self, question: str, passages: List[Dict], history: Optional[List[Dict]] = None
    ) -> Iterator[str]:
        """Diffuse la réponse par fragments (streaming).

        Implémentation par défaut (patron *Template Method*) : renvoie la réponse
        complète en un seul fragment. Les implémentations peuvent surcharger pour
        une vraie diffusion token par token.
        """
        yield self.answer(question, passages, history)
