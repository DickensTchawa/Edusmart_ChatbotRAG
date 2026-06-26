"""Script d'ingestion : construit l'index FAISS à partir du dossier docs/.

Usage (depuis le dossier app/) :
    python ingest.py
"""
import config
from services.document_service import DocumentService
from services.faiss_service import FaissService


def main():
    print("=== Ingestion des documents pédagogiques ===")
    print(f"Dossier source : {config.DOCS_DIR}")

    documents = DocumentService()
    chunks = documents.load_chunks()
    if not chunks:
        print("Aucun passage trouvé. Déposez des fichiers .pdf / .txt / .md dans docs/.")
        return

    faiss_service = FaissService()
    count = faiss_service.build_index(chunks)
    print(f"Terminé : {count} passages indexés dans {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
