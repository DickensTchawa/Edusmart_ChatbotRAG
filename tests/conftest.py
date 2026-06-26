"""Fixtures partagées et configuration des tests.

On définit un token Hugging Face factice AVANT tout import de l'application :
`LLMService` exige un token à l'instanciation (effectuée à l'import du
contrôleur). Aucun appel réseau n'est réalisé ici — le client n'est créé que
lors d'un véritable appel, mocké dans les tests.
"""
import os

os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", "test-token")

import numpy as np
import pytest


class FakeEmbedder:
    """Encodeur déterministe et hors-ligne, compatible SentenceTransformer.

    Produit un vecteur reproductible par texte (sac de caractères projeté sur
    `dim` dimensions). Suffisant pour valider l'indexation, la recherche et le
    filtrage sans télécharger le modèle réel.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for i, ch in enumerate(t.lower()):
                v[(ord(ch) + i) % self.dim] += 1.0
            # éviter le vecteur nul
            if not v.any():
                v[0] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype="float32")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_chunks():
    return [
        {"text": "La photosynthèse transforme la lumière en glucose.", "source": "bio.txt", "chunk_id": 0},
        {"text": "Le glucose est un sucre produit par les plantes.", "source": "bio.txt", "chunk_id": 1},
        {"text": "Le théorème de Pythagore relie les côtés d'un triangle rectangle.", "source": "math.txt", "chunk_id": 0},
        {"text": "Une fonction affine a la forme f(x) = ax + b.", "source": "math.txt", "chunk_id": 1},
    ]
