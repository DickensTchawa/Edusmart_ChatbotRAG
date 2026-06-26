"""Construction et interrogation de l'index vectoriel FAISS."""
import os
import json
from typing import List, Dict

import faiss
import numpy as np

import config
from services.interfaces import Retriever


class FaissService(Retriever):
    """Encapsule l'embedding (SBERT) et l'index FAISS.

    - `build_index` : (ré)génère l'index à partir d'une liste de passages.
    - `search`      : retourne les passages les plus proches d'une question.

    Le modèle d'embedding est **chargé paresseusement** (au premier usage) et
    peut être **injecté** au constructeur. Cela permet de tester le service
    hors-ligne avec un faux encodeur, sans télécharger le modèle réel, et
    facilite le remplacement de l'encodeur (principe ouvert/fermé).
    """

    def __init__(self, model_name: str = None, model=None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = model  # None => chargement paresseux
        self.index = None
        self.chunks: List[Dict] = []

    # ------------------------------------------------------------------ modèle
    @property
    def model(self):
        """Charge le modèle SentenceTransformer au premier accès."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[faiss_service] Chargement du modèle d'embedding : {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------ utils
    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.asarray(self.model.encode(texts, show_progress_bar=False)).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)  # cosinus via produit scalaire
        return vectors

    # ----------------------------------------------------------------- build
    def build_index(self, chunks: List[Dict]) -> int:
        """Construit l'index à partir des passages et l'enregistre sur le disque."""
        if not chunks:
            raise ValueError("Aucun passage à indexer. Ajoutez des fichiers dans docs/.")

        texts = [c["text"] for c in chunks]
        vectors = self._embed(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # produit scalaire sur vecteurs normalisés
        index.add(vectors)

        os.makedirs(config.INDEX_DIR, exist_ok=True)
        faiss.write_index(index, config.INDEX_PATH)
        with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        self.index = index
        self.chunks = chunks
        print(f"[faiss_service] Index construit : {len(chunks)} passages, dim={dim}.")
        return len(chunks)

    # ------------------------------------------------------------------ load
    def load_index(self) -> bool:
        """Charge l'index et les passages depuis le disque. False si absent."""
        if not (os.path.exists(config.INDEX_PATH) and os.path.exists(config.CHUNKS_PATH)):
            return False
        self.index = faiss.read_index(config.INDEX_PATH)
        with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[faiss_service] Index chargé : {len(self.chunks)} passages.")
        return True

    # ---------------------------------------------------------------- search
    def list_sources(self) -> List[str]:
        """Retourne la liste distincte des documents indexés."""
        seen = []
        for c in self.chunks:
            if c["source"] not in seen:
                seen.append(c["source"])
        return seen

    def search(self, query: str, k: int = None, sources: List[str] = None) -> List[Dict]:
        """Retourne les k passages les plus pertinents avec leur score.

        Si `sources` est fourni, la recherche est restreinte à ces documents :
        on élargit la recherche puis on filtre, afin de toujours obtenir k
        passages parmi les documents sélectionnés (idéalement de plusieurs
        sources différentes).
        """
        if self.index is None:
            if not self.load_index():
                raise RuntimeError(
                    "Index introuvable. Lancez l'ingestion (python ingest.py "
                    "ou POST /reindex) avant d'interroger le chatbot."
                )
        k = k or config.TOP_K
        n_total = len(self.chunks)
        if n_total == 0:
            return []

        # Si on filtre par sources, on récupère tous les passages puis on filtre ;
        # sinon on se limite à k pour la performance.
        search_k = n_total if sources else min(k, n_total)
        qvec = self._embed([query])
        scores, indices = self.index.search(qvec, search_k)

        source_set = set(sources) if sources else None
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[idx])
            if source_set is not None and item["source"] not in source_set:
                continue
            item["score"] = float(score)
            results.append(item)
            if len(results) >= k:
                break
        return results
