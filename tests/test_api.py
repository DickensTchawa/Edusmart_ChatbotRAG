"""Tests d'intégration de l'API (FastAPI TestClient).

Le modèle d'embedding et le LLM sont mockés : aucun téléchargement ni appel
réseau n'est effectué.
"""
import config
import main
from controllers import chat_controller
from fastapi.testclient import TestClient

client = TestClient(main.app)


def test_status_ok():
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "modele_llm" in body and "nb_passages" in body


def test_home_sert_l_interface():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_chat_question_vide_renvoie_400():
    r = client.post("/chat", json={"question": "   "})
    assert r.status_code == 400


def test_chat_reponse_mockee(monkeypatch):
    monkeypatch.setattr(
        chat_controller.faiss_service,
        "search",
        lambda q, k=None, sources=None: [
            {"text": "extrait pertinent", "source": "cours.txt", "score": 0.91}
        ],
    )
    monkeypatch.setattr(
        chat_controller.llm_service,
        "answer",
        lambda question, passages, history=None: "Réponse pédagogique de test.",
    )
    r = client.post("/chat", json={"question": "Qu'est-ce que la photosynthèse ?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Réponse pédagogique de test."
    assert "cours.txt" in body["sources"]


def test_chat_transmet_les_sources_selectionnees(monkeypatch):
    capture = {}

    def fake_search(q, k=None, sources=None):
        capture["sources"] = sources
        return [{"text": "x", "source": "math.txt", "score": 0.5}]

    monkeypatch.setattr(chat_controller.faiss_service, "search", fake_search)
    monkeypatch.setattr(
        chat_controller.llm_service, "answer", lambda question, passages, history=None: "ok"
    )

    client.post("/chat", json={"question": "test", "sources": ["math.txt"]})
    assert capture["sources"] == ["math.txt"]


def test_chat_transmet_l_historique(monkeypatch):
    capture = {}

    def fake_answer(question, passages, history=None):
        capture["history"] = history
        return "ok"

    monkeypatch.setattr(
        chat_controller.faiss_service,
        "search",
        lambda q, k=None, sources=None: [{"text": "x", "source": "a.txt", "score": 0.5}],
    )
    monkeypatch.setattr(chat_controller.llm_service, "answer", fake_answer)

    client.post(
        "/chat",
        json={
            "question": "Et sa deuxième phase ?",
            "history": [
                {"role": "user", "content": "Parle-moi de la photosynthèse"},
                {"role": "assistant", "content": "C'est un processus..."},
            ],
        },
    )
    assert capture["history"] == [
        {"role": "user", "content": "Parle-moi de la photosynthèse"},
        {"role": "assistant", "content": "C'est un processus..."},
    ]


def test_upload_et_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    # On évite la reconstruction réelle de l'index (pas d'embedding en test).
    monkeypatch.setattr(chat_controller, "_rebuild_index", lambda: 7)

    r = client.post(
        "/upload",
        files=[("files", ("cours.txt", b"contenu de cours", "text/plain"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert "cours.txt" in body["uploaded"]
    assert body["passages"] == 7

    r2 = client.get("/documents")
    assert r2.status_code == 200
    noms = [d["name"] for d in r2.json()["documents"]]
    assert "cours.txt" in noms


def test_upload_rejette_type_non_supporte(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(chat_controller, "_rebuild_index", lambda: 0)
    r = client.post(
        "/upload",
        files=[("files", ("image.png", b"\x89PNG", "image/png"))],
    )
    assert r.status_code == 400


def test_chat_stream_diffuse_tokens_et_sources(monkeypatch):
    monkeypatch.setattr(
        chat_controller.faiss_service,
        "search",
        lambda q, k=None, sources=None: [
            {"text": "extrait", "source": "cours.txt", "score": 0.9}
        ],
    )
    # Doublure de streaming : trois fragments successifs.
    monkeypatch.setattr(
        chat_controller.llm_service,
        "answer_stream",
        lambda question, passages, history=None: iter(["Bonjour", " le ", "monde"]),
    )
    with client.stream("POST", "/chat/stream", json={"question": "test"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = "".join(r.iter_text())

    # Le flux contient les sources, les tokens et l'événement de fin.
    assert '"type": "sources"' in body
    assert "cours.txt" in body
    assert "Bonjour" in body and "monde" in body
    assert '"type": "done"' in body


def test_feedback_enregistre(tmp_path, monkeypatch):
    fpath = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(chat_controller, "FEEDBACK_PATH", str(fpath))
    r = client.post(
        "/feedback",
        json={"question": "Q ?", "answer": "R.", "rating": "up"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert fpath.exists()
    assert '"rating": "up"' in fpath.read_text(encoding="utf-8")


def test_feedback_rating_invalide():
    r = client.post(
        "/feedback",
        json={"question": "Q", "answer": "R", "rating": "maybe"},
    )
    assert r.status_code == 400
