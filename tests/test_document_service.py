"""Tests du service de lecture et de découpage des documents."""
from services.document_service import DocumentService


def test_chunk_text_respecte_taille_et_chevauchement():
    ds = DocumentService(chunk_size=100, overlap=20)
    texte = "Phrase de test. " * 50  # ~800 caractères
    chunks = ds._chunk_text(texte)
    assert len(chunks) > 1
    # Aucun chunk ne dépasse nettement la taille demandée.
    assert all(len(c) <= 100 + 20 for c in chunks)


def test_chunk_text_vide_retourne_liste_vide():
    ds = DocumentService()
    assert ds._chunk_text("   ") == []


def test_load_chunks_lit_txt_et_md(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cours.txt").write_text("Le contenu du cours. " * 30, encoding="utf-8")
    (docs / "notes.md").write_text("# Titre\n\nContenu markdown.", encoding="utf-8")

    ds = DocumentService(docs_dir=str(docs), chunk_size=120, overlap=20)
    chunks = ds.load_chunks()

    assert len(chunks) > 0
    sources = {c["source"] for c in chunks}
    assert sources == {"cours.txt", "notes.md"}
    assert all({"text", "source", "chunk_id"} <= set(c) for c in chunks)


def test_load_chunks_dossier_absent_leve_erreur(tmp_path):
    ds = DocumentService(docs_dir=str(tmp_path / "inexistant"))
    try:
        ds.load_chunks()
        assert False, "FileNotFoundError attendu"
    except FileNotFoundError:
        pass
