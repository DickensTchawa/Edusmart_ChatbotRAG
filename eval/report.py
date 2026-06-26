"""Génère un rapport lisible à partir des résultats d'évaluation.

Lit `eval/results.json` (produit par `evaluate.py --json eval/results.json`) et
produit :
- `eval/eval_report.md` : tableau Markdown prêt à coller dans le mémoire ;
- `eval/eval_chart.png` : graphique comparatif (si matplotlib est installé).

Usage :
    python eval/evaluate.py --retriever both --json eval/results.json
    python eval/report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
REPORT_MD = os.path.join(HERE, "eval_report.md")
CHART_PNG = os.path.join(HERE, "eval_chart.png")

METRICS = ["hit@k", "recall@k", "precision@k", "mrr"]


def load_results(path=RESULTS):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable. Lancez d'abord :\n"
            "  python eval/evaluate.py --retriever both --json eval/results.json"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_markdown(data) -> str:
    n = data.get("n_questions", "?")
    results = data["results"]  # {mode: {k: {metric: value}}}
    lines = [
        "# Résultats d'évaluation de la récupération",
        "",
        f"Jeu de référence : **{n} questions**. Métriques moyennées par question.",
        "",
    ]
    for mode, by_k in results.items():
        lines.append(f"## Stratégie : {mode}")
        lines.append("")
        lines.append("| k | Hit@k | Recall@k | Precision@k | MRR |")
        lines.append("|---|------|---------|------------|-----|")
        for k in sorted(by_k, key=lambda x: int(x)):
            m = by_k[k]
            lines.append(
                f"| {k} | {m['hit@k']:.2f} | {m['recall@k']:.2f} | "
                f"{m['precision@k']:.2f} | {m['mrr']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_chart(data, k_focus=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib non installé — graphique ignoré "
              "(pip install matplotlib).")
        return False

    results = data["results"]
    modes = list(results)
    # Valeur de k commune à comparer (la plus grande disponible par défaut).
    ks = sorted({k for by_k in results.values() for k in by_k}, key=lambda x: int(x))
    k = str(k_focus) if k_focus is not None else ks[-1]

    teal = "#14B8C6"
    palette = [teal, "#0F172A", "#94A3B8"]
    x = range(len(METRICS))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mode in enumerate(modes):
        vals = [results[mode].get(k, {}).get(metric, 0.0) for metric in METRICS]
        offset = (i - (len(modes) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=mode, color=palette[i % len(palette)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Hit@k", "Recall@k", "Precision@k", "MRR"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (moyenne)")
    ax.set_title(f"Comparaison des stratégies de récupération (k = {k})")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHART_PNG, dpi=160)
    print(f"[report] Graphique enregistré : {CHART_PNG}")
    return True


def main():
    data = load_results()
    md = build_markdown(data)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] Tableau Markdown enregistré : {REPORT_MD}\n")
    print(md)
    build_chart(data)


if __name__ == "__main__":
    main()
