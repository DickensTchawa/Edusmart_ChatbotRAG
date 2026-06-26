"""Génération de la réponse pédagogique via l'API d'inférence Hugging Face.

Robustesse : le routeur Hugging Face n'expose pas tous les modèles en mode
« chat » chez tous les providers. Pour éviter les erreurs « not supported by
provider », le service essaie le provider configuré puis bascule
automatiquement sur d'autres providers jusqu'à trouver une combinaison qui
fonctionne (résultat mis en cache pour les appels suivants).
"""
from typing import Dict, Iterator, List

from huggingface_hub import InferenceClient

import config
from services.interfaces import LLMProvider


SYSTEM_PROMPT = (
    "Tu es un tuteur pédagogue, clair et bienveillant. "
    "Tu réponds aux questions d'un apprenant en t'appuyant UNIQUEMENT sur le "
    "contexte fourni (extraits de cours). "
    "Règles :\n"
    "1. Si le contexte ne contient pas la réponse, dis-le honnêtement et "
    "n'invente rien.\n"
    "2. Explique simplement, avec des exemples si utile, comme à un étudiant.\n"
    "3. Structure ta réponse et reste concis.\n"
    "4. Réponds dans la langue de la question (français par défaut)."
)

# Providers tentés en secours (après celui configuré), si le modèle n'est pas
# servi en mode chat. Ordre = du plus probable au moins probable.
FALLBACK_PROVIDERS = [
    "auto", "together", "fireworks-ai", "nebius",
    "hyperbolic", "sambanova", "novita", "hf-inference",
]

# Erreurs de routage que l'on peut résoudre en changeant de provider.
_ROUTING_ERRORS = ("not supported", "not a chat model", "no provider", "not found")


class LLMService(LLMProvider):
    """Appelle un modèle instruct ouvert via l'API d'inférence Hugging Face."""

    def __init__(self, token: str = None, model: str = None, provider: str = None):
        self.token = token or config.HUGGINGFACEHUB_API_TOKEN
        self.model = model or config.LLM_MODEL
        self.provider = provider if provider is not None else config.LLM_PROVIDER
        if not self.token:
            raise ValueError(
                "Le token d'API Hugging Face est requis. "
                "Renseignez HUGGINGFACEHUB_API_TOKEN dans app/.env."
            )
        # Provider qui a fonctionné lors d'un appel précédent (cache).
        self._working_provider = None
        print(f"[llm_service] Modèle : {self.model} "
              f"(provider configuré : {self.provider or 'auto'})")

    def _provider_sequence(self) -> List[str]:
        """Ordre des providers à essayer : configuré d'abord, puis secours."""
        seq = []
        first = self._working_provider or self.provider or "auto"
        seq.append(first)
        for p in FALLBACK_PROVIDERS:
            if p not in seq:
                seq.append(p)
        return seq

    @staticmethod
    def _build_context(passages: List[Dict]) -> str:
        blocks = []
        for i, p in enumerate(passages, 1):
            src = p.get("source", "?")
            blocks.append(f"[Extrait {i} — source : {src}]\n{p['text']}")
        return "\n\n".join(blocks)

    _NO_CONTEXT = (
        "Je n'ai trouvé aucun passage pertinent dans les documents pour "
        "répondre à cette question."
    )

    MAX_HISTORY = 6  # nombre maximal de messages d'historique conservés

    def _messages(
        self, question: str, passages: List[Dict], history: List[Dict] = None
    ) -> List[Dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Tours précédents (questions de suivi), bornés pour limiter la taille du prompt.
        for m in (history or [])[-self.MAX_HISTORY:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        context = self._build_context(passages)
        user_message = (
            f"Contexte (extraits de cours) :\n{context}\n\n"
            f"Question de l'apprenant : {question}\n\n"
            "Réponds en t'appuyant uniquement sur le contexte ci-dessus."
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    def answer(
        self, question: str, passages: List[Dict], history: List[Dict] = None
    ) -> str:
        """Génère une réponse en langage naturel à partir des passages."""
        if not passages:
            return self._NO_CONTEXT

        messages = self._messages(question, passages, history)

        last_error = "aucun provider disponible"
        for prov in self._provider_sequence():
            try:
                client = InferenceClient(model=self.model, token=self.token, provider=prov)
                completion = client.chat_completion(
                    messages=messages, max_tokens=512, temperature=0.2,
                )
                self._working_provider = prov  # mémoriser pour la prochaine fois
                print(f"[llm_service] Réponse obtenue via provider : {prov}")
                return completion.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                if any(k in last_error.lower() for k in _ROUTING_ERRORS):
                    print(f"[llm_service] provider '{prov}' inadapté, essai suivant…")
                    continue
                # Erreur non liée au routage (auth, quota, réseau) : inutile d'insister.
                break

        return self._error_message(last_error)

    def _error_message(self, last_error: str) -> str:
        return (
            "Impossible d'obtenir une réponse du modèle Hugging Face. "
            f"Dernière erreur : {last_error}\n\n"
            f"Le modèle « {self.model} » n'est peut-être servi en mode chat par "
            "aucun de vos providers activés. Solutions : choisissez un autre modèle "
            "instruct (LLM_MODEL dans app/.env), ou activez des providers (gratuit) "
            "sur https://huggingface.co/settings/inference-providers"
        )

    def answer_stream(
        self, question: str, passages: List[Dict], history: List[Dict] = None
    ) -> Iterator[str]:
        """Diffuse la réponse token par token (streaming) via Hugging Face.

        Conserve le même mécanisme de repli sur les providers que `answer` :
        l'erreur de routage survient à la première itération (avant tout
        fragment émis), ce qui permet de basculer sans réponse partielle.
        """
        if not passages:
            yield self._NO_CONTEXT
            return

        messages = self._messages(question, passages, history)
        last_error = "aucun provider disponible"
        for prov in self._provider_sequence():
            try:
                client = InferenceClient(model=self.model, token=self.token, provider=prov)
                stream = client.chat_completion(
                    messages=messages, max_tokens=512, temperature=0.2, stream=True,
                )
                emitted = False
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    self._working_provider = prov
                    return
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                if any(k in last_error.lower() for k in _ROUTING_ERRORS):
                    continue
                break

        yield self._error_message(last_error)
