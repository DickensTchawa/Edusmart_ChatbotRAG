"""Endpoints du chatbot : interrogation, téléversement, réindexation."""
import json
import os
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from services.document_service import DocumentService
from services.faiss_service import FaissService
from services.interfaces import LLMProvider, Retriever
from services.llm_service import LLMService

router = APIRouter()

# Services initialisés une seule fois au démarrage.
# Annotés par leurs interfaces : le contrôleur dépend des abstractions
# (Retriever / LLMProvider), pas des implémentations concrètes — ce qui rend
# FAISS et le fournisseur de LLM remplaçables sans toucher à ce fichier.
faiss_service: Retriever = FaissService()
llm_service: LLMProvider = LLMService()
document_service = DocumentService()

# Tenter de charger un index déjà construit.
faiss_service.load_index()


def _make_retriever() -> Retriever:
    """Construit le retriever actif selon `config.RETRIEVER`.

    - "dense"  : FAISS seul (retourne l'objet faiss_service lui-même).
    - "hybrid" : fusion FAISS + BM25 (+ re-ranking optionnel), interchangeable
      sans modifier les endpoints car HybridRetriever respecte l'interface.
    """
    if config.RETRIEVER == "hybrid":
        from services.bm25_retriever import BM25Retriever
        from services.hybrid_retriever import HybridRetriever

        reranker = None
        if config.RERANK:
            from services.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker()
        lexical = BM25Retriever(getattr(faiss_service, "chunks", []))
        print(f"[retriever] Mode hybride (BM25 + dense), re-ranking={config.RERANK}")
        return HybridRetriever(faiss_service, lexical, reranker=reranker)
    return faiss_service


# Retriever effectivement utilisé par les endpoints.
retriever: Retriever = _make_retriever()

ALLOWED_EXT = (".pdf", ".txt", ".md")


class Message(BaseModel):
    role: str       # "user" ou "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    sources: List[str] | None = None  # documents sur lesquels restreindre la recherche
    history: List[Message] | None = None  # tours précédents (mémoire conversationnelle)


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "up" ou "down"


FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedback.jsonl")


def _unique_sources(passages: List[dict]) -> List[str]:
    out: List[str] = []
    for p in passages:
        if p["source"] not in out:
            out.append(p["source"])
    return out


def _history_dicts(request: "ChatRequest") -> List[dict]:
    """Convertit l'historique (modèles Pydantic) en liste de dictionnaires."""
    if not request.history:
        return []
    return [{"role": m.role, "content": m.content} for m in request.history]


def _rebuild_index() -> int:
    global retriever
    chunks = document_service.load_chunks()
    n = faiss_service.build_index(chunks)
    retriever = _make_retriever()  # reconstruit aussi l'index lexical en mode hybride
    return n


@router.post("/chat")
async def chat(request: ChatRequest):
    """Répond à une question à partir des documents indexés."""
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")

    try:
        passages = retriever.search(question, k=request.top_k, sources=request.sources)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    answer = llm_service.answer(question, passages, history=_history_dicts(request))
    sources = _unique_sources(passages)

    return {
        "answer": answer,
        "sources": sources,
        "passages": [
            {"source": p["source"], "score": round(p["score"], 3), "extrait": p["text"][:300]}
            for p in passages
        ],
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Diffuse la réponse en flux (Server-Sent Events) pour un rendu temps réel.

    Format des événements : lignes `data: {json}` séparées par une ligne vide.
    - {"type":"sources","sources":[...]}   (une fois, au début)
    - {"type":"token","text":"..."}        (pour chaque fragment)
    - {"type":"done"}                      (fin)
    """
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")

    try:
        passages = retriever.search(question, k=request.top_k, sources=request.sources)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    sources = _unique_sources(passages)
    history = _history_dicts(request)

    def event_stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield sse({"type": "sources", "sources": sources})
        for delta in llm_service.answer_stream(question, passages, history=history):
            yield sse({"type": "token", "text": delta})
        yield sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Enregistre un retour utilisateur (👍/👎) dans feedback.jsonl."""
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating doit être 'up' ou 'down'.")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": request.rating,
        "question": request.question,
        "answer": request.answer,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Téléverse un ou plusieurs documents dans docs/ puis reconstruit l'index."""
    os.makedirs(os.path.abspath(config.DOCS_DIR), exist_ok=True)
    saved, ignored = [], []

    for f in files:
        name = os.path.basename(f.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in ALLOWED_EXT:
            ignored.append(f.filename)
            continue
        dest = os.path.join(os.path.abspath(config.DOCS_DIR), name)
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(name)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier valide. Formats acceptés : .pdf, .txt, .md",
        )

    try:
        count = _rebuild_index()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Index non reconstruit : {e}")

    return {"uploaded": saved, "ignored": ignored, "passages": count}


@router.get("/documents")
async def documents():
    """Liste les documents présents dans docs/."""
    d = os.path.abspath(config.DOCS_DIR)
    items = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in ALLOWED_EXT:
                items.append({"name": name, "size_kb": round(os.path.getsize(p) / 1024, 1)})
    return {"documents": items, "count": len(items)}


@router.post("/reindex")
async def reindex():
    """(Re)construit l'index FAISS à partir des fichiers du dossier docs/."""
    try:
        count = _rebuild_index()
        return {"message": f"Index reconstruit avec succès ({count} passages).",
                "documents_dir": config.DOCS_DIR}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def status():
    """État de l'index."""
    loaded = faiss_service.index is not None
    return {
        "index_charge": loaded,
        "nb_passages": len(faiss_service.chunks) if loaded else 0,
        "modele_llm": config.LLM_MODEL,
        "modele_embedding": config.EMBEDDING_MODEL,
    }
