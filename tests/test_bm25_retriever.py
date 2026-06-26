"""Tests du retriever lexical BM25 (nécessite rank_bm25)."""
import pytest

pytest.importorskip("rank_bm25")

from services.bm25_retriever import BM25Retriever  # noqa: E402


def test_bm25_classe_le_document_pertinent_en_tete(sample_chunks):
    r = BM25Retriever(sample_chunks)
    res = r.search("théorème de Pythagore triangle rectangle", k=2)
    assert res
    assert res[0]["source"] == "math.txt"


def test_bm25_filtre_par_sources(sample_chunks):
    r = BM25Retriever(sample_chunks)
    res = r.search("glucose plantes", k=5, sources=["bio.txt"])
    assert res
    assert all(p["source"] == "bio.txt" for p in res)


def test_bm25_corpus_vide_renvoie_liste_vide():
    assert BM25Retriever([]).search("quoi que ce soit") == []


def test_bm25_est_un_retriever(sample_chunks):
    from services.interfaces import Retriever

    assert isinstance(BM25Retriever(sample_chunks), Retriever)
