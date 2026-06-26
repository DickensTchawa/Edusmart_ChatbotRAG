"""Point d'entrée FastAPI du chatbot RAG pédagogique.

Sert l'API (/chat, /reindex, /status) et une interface de chat web (/).
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config
from controllers.chat_controller import router as chat_router

app = FastAPI(title="Chatbot RAG pédagogique", version="1.0")
app.include_router(chat_router)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    """Sert l'interface de chat."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    if not config.HUGGINGFACEHUB_API_TOKEN:
        raise RuntimeError(
            "Le token d'API Hugging Face n'est pas défini. "
            "Renseignez HUGGINGFACEHUB_API_TOKEN dans app/.env."
        )
    print("Démarrage du serveur sur http://localhost:8002 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
