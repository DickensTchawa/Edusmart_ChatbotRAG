"""Évaluation chiffrée de la récupération du chatbot RAG.

Indexe le corpus de référence (eval/corpus/), exécute chaque question du jeu
de données (eval/dataset.jsonl) et calcule les métriques de récupération
(Hit@k, Recall@k, Precision@k, MRR) pour une ou plusieurs valeurs de k.

Usage :
    python eval/evaluate.py                 # k = 1, 3, 5
    python eval/evaluate.py --k 3            # une seule valeur
    python eval/evaluate.py --json out.json # sauvegarde des résultats

Remarque : ce script charge le vrai modèle d'embedding (téléchargé au premier
lancement). Les métriques, elles, sont testées séparément et hors-ligne
(tests/test_eval_metrics.py).
"""
import argparse
import json
import os
import sys

# Rendre importables les modules de app/ et eval/.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, HERE)

from metrics import aggregate, evaluate_query  # noqa: E402

CORPUS_DIR = os.path.join(HERE, "corpus")
DATASET = os.path.join(HERE, "dataset.jsonl")


def _dedupe(seq):
    """Déduplique en préservant l'ordre (sources de chunks -> sources uniques)."""
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_dataset(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_retriever(mode="dense"):
    """Construit le retriever (dense ou hybride) indexé sur le corpus de référence.

    L'index est écrit dans un dossier dédié (eval/.index_eval) afin de ne PAS
    écraser l'index de production (app/index_store).
    """
    import config
    from services.document_service import DocumentService
    from services.faiss_service import FaissService

    eval_index = os.path.join(HERE, ".index_eval")
    config.INDEX_DIR = eval_index
    config.INDEX_PATH = os.path.join(eval_index, "faiss.index")
    config.CHUNKS_PATH = os.path.join(eval_index, "chunks.json")

    # Découpage plus fin pour l'évaluation : plusieurs passages par document,
    # ce qui rend la récupération non triviale (utile pour comparer les stratégies).
    chunks = DocumentService(docs_dir=CORPUS_DIR, chunk_size=350, overlap=60).load_chunks()
    dense = FaissService()
    dense.build_index(chunks)

    if mode == "hybrid":
        from services.bm25_retriever import BM25Retriever
        from services.hybrid_retriever import HybridRetriever

        return HybridRetriever(dense, BM25Retriever(chunks))
    return dense


def run(ks, dataset_path=DATASET, retriever=None, mode="dense"):
    """Exécute l'évaluation et retourne {k: metriques_agregees}."""
    data = load_dataset(dataset_path)
    retriever = retriever or build_retriever(mode)
    results = {}
    for k in ks:
        per_query = []
        for row in data:
            passages = retriever.search(row["question"], k=k)
            retrieved = _dedupe([p["source"] for p in passages])
            per_query.append(evaluate_query(retrieved, row["expected_sources"], k))
        results[k] = aggregate(per_query)
    return results, len(data)


def print_table(results, n, mode="dense"):
    print(f"\nÉvaluation de la récupération [{mode}] — {n} questions de référence\n")
    header = f"{'k':>3} | {'Hit@k':>7} | {'Recall@k':>9} | {'Precision@k':>12} | {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for k, m in sorted(results.items()):
        print(f"{k:>3} | {m['hit@k']:>7.2f} | {m['recall@k']:>9.2f} | "
              f"{m['precision@k']:>12.2f} | {m['mrr']:>6.2f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Évaluation RAG (récupération).")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 3, 5],
                    help="Valeurs de k à évaluer (défaut : 1 3 5).")
    ap.add_argument("--retriever", choices=["dense", "hybrid", "both"], default="dense",
                    help="Stratégie à évaluer (défaut : dense ; 'both' compare les deux).")
    ap.add_argument("--json", type=str, default=None, help="Fichier de sortie JSON.")
    args = ap.parse_args()

    modes = ["dense", "hybrid"] if args.retriever == "both" else [args.retriever]
    all_results = {}
    n = 0
    for mode in modes:
        results, n = run(args.k, mode=mode)
        print_table(results, n, mode)
        all_results[mode] = results

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"n_questions": n, "results": all_results}, f, ensure_ascii=False, indent=2)
        print(f"Résultats enregistrés dans {args.json}")


if __name__ == "__main__":
    main()
