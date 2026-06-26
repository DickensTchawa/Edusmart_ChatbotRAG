"""Tests de l'index vectoriel : construction, recherche, filtrage par sources."""
import config
from services.faiss_service import FaissService


def _service(tmp_path, fake_embedder, monkeypatch):
    """Instancie un FaissService avec encodeur factice et index en zone temporaire."""
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(config, "INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(config, "CHUNKS_PATH", str(tmp_path / "chunks.json"))
    return FaissService(model=fake_embedder)


def test_build_index_cree_les_fichiers(tmp_path, fake_embedder, sample_chunks, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    n = fs.build_index(sample_chunks)
    assert n == len(sample_chunks)
    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "chunks.json").exists()


def test_build_index_vide_leve_erreur(tmp_path, fake_embedder, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    try:
        fs.build_index([])
        assert False, "ValueError attendu"
    except ValueError:
        pass


def test_search_retourne_des_passages(tmp_path, fake_embedder, sample_chunks, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    fs.build_index(sample_chunks)
    res = fs.search("glucose photosynthèse", k=2)
    assert 1 <= len(res) <= 2
    assert all("score" in r and "source" in r for r in res)


def test_search_filtre_par_sources(tmp_path, fake_embedder, sample_chunks, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    fs.build_index(sample_chunks)
    res = fs.search("triangle rectangle", k=5, sources=["math.txt"])
    assert res, "au moins un passage attendu"
    assert all(r["source"] == "math.txt" for r in res)


def test_list_sources(tmp_path, fake_embedder, sample_chunks, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    fs.build_index(sample_chunks)
    assert set(fs.list_sources()) == {"bio.txt", "math.txt"}


def test_load_index_recharge_depuis_disque(tmp_path, fake_embedder, sample_chunks, monkeypatch):
    fs = _service(tmp_path, fake_embedder, monkeypatch)
    fs.build_index(sample_chunks)
    fs2 = _service(tmp_path, fake_embedder, monkeypatch)
    assert fs2.load_index() is True
    assert len(fs2.chunks) == len(sample_chunks)
