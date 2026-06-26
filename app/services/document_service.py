"""Chargement et découpage des documents pédagogiques (PDF, TXT, MD)."""
import os
import glob
from typing import List, Dict

import config


class DocumentService:
    """Lit les fichiers du dossier docs/ et les découpe en passages (chunks)."""

    SUPPORTED = (".pdf", ".txt", ".md")

    def __init__(self, docs_dir: str = None, chunk_size: int = None, overlap: int = None):
        self.docs_dir = docs_dir or config.DOCS_DIR
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP

    # ------------------------------------------------------------------ lecture
    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        # Import local pour éviter de dépendre de pypdf si aucun PDF n'est utilisé.
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)

    def _read_file(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(path)
        return self._read_txt(path)

    # --------------------------------------------------------------- découpage
    def _chunk_text(self, text: str) -> List[str]:
        """Découpe en blocs de `chunk_size` caractères avec chevauchement.

        On essaie de couper sur un saut de ligne ou un espace proche de la fin
        du bloc pour ne pas trancher au milieu d'un mot.
        """
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                # Reculer jusqu'à un séparateur naturel s'il existe dans la marge.
                window = text[start:end]
                cut = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
                if cut > self.chunk_size * 0.5:
                    end = start + cut + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(end - self.overlap, start + 1)
        return chunks

    # ------------------------------------------------------------------ public
    def load_chunks(self) -> List[Dict]:
        """Retourne une liste de passages : {text, source, chunk_id}."""
        docs_dir = os.path.abspath(self.docs_dir)
        if not os.path.isdir(docs_dir):
            raise FileNotFoundError(f"Dossier de documents introuvable : {docs_dir}")

        results: List[Dict] = []
        files = []
        for ext in self.SUPPORTED:
            files.extend(glob.glob(os.path.join(docs_dir, "**", f"*{ext}"), recursive=True))
        files = sorted(set(files))

        for path in files:
            try:
                raw = self._read_file(path)
            except Exception as e:  # noqa: BLE001
                print(f"[document_service] Échec de lecture de {path} : {e}")
                continue
            source = os.path.relpath(path, docs_dir)
            for i, chunk in enumerate(self._chunk_text(raw)):
                results.append({"text": chunk, "source": source, "chunk_id": i})

        print(f"[document_service] {len(files)} fichier(s) -> {len(results)} passage(s).")
        return results
