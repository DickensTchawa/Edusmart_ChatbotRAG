"""Configuration centrale du chatbot RAG pédagogique.

Toutes les valeurs sont lues depuis les variables d'environnement (fichier .env).
"""
import os
from dotenv import load_dotenv

# Charger le fichier .env situé à côté de ce module (app/.env)
load_dotenv()

# --- Hugging Face ---
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Modèle de génération (LLM) servi par l'API d'inférence Hugging Face.
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Provider d'inférence Hugging Face.
# "auto" => HF choisit automatiquement un provider activé sur votre compte qui
# sert ce modèle en mode chat. Vous pouvez forcer un provider précis
# (ex. "nebius", "together", "fireworks-ai", "hf-inference") si besoin.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# Modèle d'embedding (vectorisation) exécuté localement via sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chemins ---
# Dossier où l'utilisateur dépose ses documents (PDF / TXT / MD).
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(os.path.dirname(__file__), "..", "docs"))
# Emplacement de l'index FAISS et de ses métadonnées.
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(os.path.dirname(__file__), "index_store"))
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# --- Paramètres de découpage (chunking) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # caractères par chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # chevauchement entre chunks

# --- Paramètres de recherche ---
TOP_K = int(os.getenv("TOP_K", "4"))  # nombre de passages récupérés par question

# Stratégie de récupération : "dense" (FAISS seul) ou "hybrid" (FAISS + BM25 fusionnés).
RETRIEVER = os.getenv("RETRIEVER", "dense").lower()
# Nombre de candidats demandés à chaque sous-retriever avant fusion (mode hybride).
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "10"))

# Re-ranking par cross-encoder (mode hybride). Désactivé par défaut.
RERANK = os.getenv("RERANK", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
