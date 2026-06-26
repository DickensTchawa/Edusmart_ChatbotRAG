"""Tests des abstractions Retriever / LLMProvider (principe d'inversion + Liskov).

Démontre que :
- les implémentations réelles respectent les contrats ;
- des doublures interchangeables peuvent les remplacer ;
- une implémentation incomplète est rejetée (méthode abstraite manquante).
"""
from typing import Dict, List, Optional

import pytest

from services.faiss_service import FaissService
from services.interfaces import LLMProvider, Retriever
from services.llm_service import LLMService


def test_faiss_service_est_un_retriever(fake_embedder):
    fs = FaissService(model=fake_embedder)
    assert isinstance(fs, Retriever)


def test_llm_service_est_un_llmprovider():
    assert isinstance(LLMService(), LLMProvider)


class FakeRetriever(Retriever):
    def search(self, query: str, k: Optional[int] = None, sources: Optional[List[str]] = None) -> List[Dict]:
        return [{"text": "passage factice", "source": "x.txt", "score": 1.0}]

    def list_sources(self) -> List[str]:
        return ["x.txt"]


class FakeLLM(LLMProvider):
    def answer(self, question: str, passages: List[Dict], history=None) -> str:
        return "réponse factice"


def test_implementations_sont_substituables():
    retriever: Retriever = FakeRetriever()
    llm: LLMProvider = FakeLLM()
    passages = retriever.search("question")
    assert passages and passages[0]["source"] == "x.txt"
    assert llm.answer("question", passages) == "réponse factice"


def test_implementation_incomplete_est_refusee():
    # Méthode abstraite list_sources manquante -> instanciation impossible.
    class RetrieverIncomplet(Retriever):
        def search(self, query, k=None, sources=None):
            return []

    with pytest.raises(TypeError):
        RetrieverIncomplet()
