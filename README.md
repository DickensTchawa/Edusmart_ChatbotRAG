# Chatbot RAG pédagogique

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)

> Remplacez `OWNER/REPO` dans l'URL du badge ci-dessus par votre dépôt GitHub
> (ex. `dickenstchawa/rag-pedagogique`) pour afficher l'état réel de la CI.

Assistant de cours qui répond aux questions d'un apprenant **à partir de vos
propres documents** (PDF, TXT, MD). Inspiré de l'architecture du projet AI ONA
BTP, mais la source de connaissance n'est plus Odoo : ce sont des documents que
vous indexez.

## Pipeline

```
Documents (docs/)                    Question de l'apprenant
      │                                       │
      ▼                                       ▼
DocumentService  ──►  FaissService  ◄──  embedding de la question
(lecture + chunking)   (index vectoriel)        │
                              │ top-k passages   │
                              ▼                   │
                         LLMService (Hugging Face, Mistral-7B-Instruct)
                              │
                              ▼
                     Réponse pédagogique + sources
```

- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (local).
- **Recherche** : FAISS (`IndexFlatIP`, similarité cosinus).
- **Génération** : modèle ouvert via l'API d'inférence Hugging Face
  (`mistralai/Mistral-7B-Instruct-v0.3` par défaut).
- **Interface** : API FastAPI + page de chat web.

## Arborescence

```
rag-pedagogique/
├── docs/                       ← déposez ici vos PDF / TXT / MD
│   └── exemple_photosynthese.txt
├── app/
│   ├── .env                    ← votre token Hugging Face
│   ├── config.py
│   ├── ingest.py               ← construit l'index
│   ├── main.py                 ← API + UI (port 8002)
│   ├── controllers/chat_controller.py
│   ├── services/
│   │   ├── document_service.py
│   │   ├── faiss_service.py
│   │   └── llm_service.py
│   └── static/index.html       ← interface de chat
├── requirements.txt
├── run.bat
└── README.md
```

## Démarrage (Windows)

1. Renseignez `HUGGINGFACEHUB_API_TOKEN` dans `app\.env`
   (token sur https://huggingface.co/settings/tokens).
2. Déposez vos documents dans `docs\` (un exemple est déjà fourni).
3. Double-cliquez sur **`run.bat`**.

Le script crée le venv, installe les dépendances, construit l'index, puis lance
le serveur. Ouvrez ensuite **http://localhost:8002**.

### Démarrage manuel

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd app
python ingest.py        REM construit l'index depuis docs/
python main.py          REM lance le serveur
```

## API

| Méthode | Route        | Description                                    |
|---------|--------------|------------------------------------------------|
| GET     | `/`          | Interface de chat web                          |
| POST    | `/chat`      | `{ "question": "..." }` → réponse + sources    |
| POST    | `/chat/stream` | Réponse en flux (SSE), affichée token par token |
| POST    | `/feedback`  | Enregistre un retour 👍/👎 sur une réponse      |
| POST    | `/reindex`   | Reconstruit l'index depuis `docs/`             |
| GET     | `/status`    | État de l'index et modèles utilisés            |
| GET     | `/docs`      | Documentation interactive Swagger              |

Exemple :

```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Quelles sont les deux phases de la photosynthèse ?\"}"
```

## Ajouter / mettre à jour des documents

Déposez de nouveaux fichiers dans `docs\`, puis relancez l'ingestion
(`python ingest.py`) ou appelez `POST /reindex`. L'index est stocké dans
`app\index_store\`.

## Tests, qualité & CI

Le projet est couvert par une suite de tests `pytest` (17 tests : découpage des
documents, index FAISS et filtrage par sources, endpoints de l'API). Les tests
sont **hors-ligne** : le modèle d'embedding et le LLM sont remplacés par des
doublures, donc aucun téléchargement ni appel réseau.

```bash
# Installer les dépendances de développement
pip install -r requirements-dev.txt

# Lancer les tests
pytest

# Qualité de code
ruff check app tests        # lint
black app tests             # formatage
mypy app                    # vérification de types

# Activer les hooks pre-commit (lint/format avant chaque commit)
pre-commit install
```

Une **intégration continue** (GitHub Actions, `.github/workflows/ci.yml`) exécute
les tests et les contrôles de qualité à chaque *push* et *pull request*.

## Évaluation chiffrée du RAG

Un cadre d'évaluation mesure la qualité de la **récupération** (le retriever),
sur un corpus de référence (`eval/corpus/`) et un jeu de questions annotées
(`eval/dataset.jsonl` : question, documents attendus, réponse de référence).

```bash
python eval/evaluate.py                      # mode dense, k = 1, 3, 5
python eval/evaluate.py --retriever both      # compare dense vs hybride (avant/après)
python eval/evaluate.py --k 3 --json eval/results.json
```

> **Stratégie de récupération.** Par défaut, la recherche est *dense* (FAISS).
> En passant `RETRIEVER=hybrid` (dans `app/.env`), l'application combine FAISS et
> **BM25** par *Reciprocal Rank Fusion*, avec un **re-ranking** optionnel par
> cross-encoder (`RERANK=true`). Ces stratégies sont des implémentations
> interchangeables de l'interface `Retriever`.

Métriques calculées (moyennées sur les questions) :

| Métrique | Sens |
|---|---|
| **Hit@k** | au moins un bon document dans le top-k |
| **Recall@k** | proportion des bons documents retrouvés |
| **Precision@k** | proportion de bons documents parmi les k retournés |
| **MRR** | rang inverse du premier bon document |

Exemple de sortie :

```
  k |   Hit@k |  Recall@k |  Precision@k |    MRR
----------------------------------------------------
  1 |    ...  |    ...    |     ...      |   ...
  3 |    ...  |    ...    |     ...      |   ...
  5 |    ...  |    ...    |     ...      |   ...
```

Les fonctions de métriques (`eval/metrics.py`) sont **testées hors-ligne**
(`tests/test_eval_metrics.py`). Pour évaluer la **qualité des réponses**
(fidélité, pertinence), on peut étendre l'évaluation avec un juge LLM (ex.
bibliothèque RAGAS) — c'est une perspective d'amélioration.

## Docker

```bash
docker compose up --build      # démarre le chatbot sur http://localhost:8002
```

Les documents (`docs/`) et l'index sont montés en volumes pour persister entre
les redémarrages. Le token Hugging Face est lu depuis `app/.env`.

## Architecture (extrait)

Le code suit une architecture en couches et programme **par interfaces** : les
abstractions `Retriever` et `LLMProvider` (dans `app/services/interfaces.py`)
définissent les contrats, et `FaissService` / `LLMService` en sont des
implémentations interchangeables (patrons *Strategy* / *Adapter*, principe
d'inversion des dépendances). On peut ainsi remplacer FAISS ou le fournisseur de
LLM sans modifier le contrôleur.

## Remarques

- Le modèle d'embedding (~80 Mo) est téléchargé au premier lancement.
- Si le modèle LLM choisi n'est pas disponible sur l'API d'inférence, changez
  `LLM_MODEL` dans `app\.env` pour un autre modèle instruct ouvert.
