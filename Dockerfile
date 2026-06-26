# Image légère et reproductible pour le chatbot RAG pédagogique.
FROM python:3.12-slim

WORKDIR /app

# Cache Hugging Face dans un emplacement inscriptible.
ENV HF_HOME=/app/.cache/huggingface \
    DOCS_DIR=/data/docs \
    INDEX_DIR=/data/index_store \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du contenu de app/ vers /app (main.py, services/, controllers/, static/).
COPY app/ .

EXPOSE 8002

# Démarrage du serveur ASGI.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
