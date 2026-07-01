"""Geradores de plots para avaliação de modelos de recomendação.

Fornece funções padronizadas para salvar gráficos de métricas,
popularidade de itens e cobertura de catálogo como artefatos
do MLflow. Usa matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_metrics_bar_chart(
    metrics: dict[str, float],
    output_path: str | Path,
    title: str = "Recommender Metrics",
) -> None:
    """Salva gráfico de barras com métricas de recomendação.

    Args:
        metrics: Dicionário com nome da métrica e valor float.
        output_path: Caminho para salvar a imagem (PNG).
        title: Título do gráfico.
    """
    if not metrics:
        return

    sorted_metrics = dict(sorted(metrics.items(), key=lambda x: x[1], reverse=True))
    names = list(sorted_metrics.keys())
    values = list(sorted_metrics.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color="#4c72b0")
    ax.set_xlabel("Valor")
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_at_k_curve(
    metrics_by_k: dict[int, dict[str, float]],
    output_path: str | Path,
    title: str = "Precision & Recall vs K",
) -> None:
    """Salva curva de Precision e Recall em função de K.

    Args:
        metrics_by_k: Dicionário mapeando K para métricas.
            Exemplo: {5: {"precision@5": 0.4, "recall@5": 0.2}, ...}
        output_path: Caminho para salvar a imagem (PNG).
        title: Título do gráfico.
    """
    k_values = sorted(metrics_by_k.keys())
    if not k_values:
        return

    precisions = [metrics_by_k[k].get(f"precision@{k}", 0.0) for k in k_values]
    recalls = [metrics_by_k[k].get(f"recall@{k}", 0.0) for k in k_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, precisions, "o-", label="Precision@K", color="#4c72b0")
    ax.plot(k_values, recalls, "s--", label="Recall@K", color="#dd8452")
    ax.set_xlabel("K (Top-K recomendados)")
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_item_popularity_distribution(
    item_counts: dict[str, int],
    output_path: str | Path,
    top_n: int = 30,
    title: str = "Item Popularity Distribution",
) -> None:
    """Salva histograma de popularidade dos itens.

    Args:
        item_counts: Dicionário mapeando item_id para contagem
            de interações.
        output_path: Caminho para salvar a imagem (PNG).
        top_n: Número de itens mais populares a destacar.
        title: Título do gráfico.
    """
    if not item_counts:
        return

    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    names = [item_id for item_id, _ in top_items]
    counts = [count for _, count in top_items]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), counts, color="#55a868", alpha=0.8)
    ax.set_xlabel("Item")
    ax.set_ylabel("Número de interações")
    ax.set_title(title)

    if len(names) <= 15:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks([])

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_catalog_coverage_plot(
    coverage_by_k: dict[int, float],
    output_path: str | Path,
    total_items: int,
    title: str = "Catalog Coverage vs K",
) -> None:
    """Salva gráfico de cobertura do catálogo por K.

    Args:
        coverage_by_k: Dicionário mapeando K para fração de
            itens do catálogo cobertos.
        output_path: Caminho para salvar a imagem (PNG).
        total_items: Número total de itens no catálogo.
        title: Título do gráfico.
    """
    if not coverage_by_k:
        return

    k_values = sorted(coverage_by_k.keys())
    coverages = [coverage_by_k[k] for k in k_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, coverages, "o-", color="#c44e52", label="Cobertura")
    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Cobertura total",
    )
    ax.set_xlabel("K (Top-K recomendados)")
    ax.set_ylabel("Fração do catálogo coberta")
    ax.set_title(f"{title} (catálogo: {total_items} itens)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_chart(
    all_metrics: dict[str, dict[str, float]],
    metric_keys: list[str] | None = None,
    output_path: str | Path = "reports/model_comparison.png",
    title: str = "Model Comparison",
) -> None:
    """Salva gráfico de barras agrupadas comparando modelos.

    Args:
        all_metrics: Dicionário mapeando nome do modelo para
            dicionário de métricas.
            Exemplo: {"popularity": {"precision@5": 0.1}, ...}
        metric_keys: Lista de chaves de métricas para comparar.
            Se None, usa todas as chaves encontradas.
        output_path: Caminho para salvar a imagem (PNG).
        title: Título do gráfico.
    """
    if not all_metrics:
        return

    if metric_keys is None:
        metric_keys = list(
            dict.fromkeys(
                k
                for model_metrics in all_metrics.values()
                for k in model_metrics
                if k != "num_users"
            )
        )

    model_names = list(all_metrics.keys())
    n_metrics = len(metric_keys)
    n_models = len(model_names)

    x = list(range(n_metrics))
    width = 0.8 / max(n_models, 1)

    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        model_values = [all_metrics[model_name].get(k, 0.0) for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        color = colors[i % len(colors)]
        ax.bar(
            [xi + offset for xi in x],
            model_values,
            width=width,
            label=model_name,
            color=color,
        )

    ax.set_xlabel("Métricas")
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
