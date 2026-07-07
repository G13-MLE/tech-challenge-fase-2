"""Gerador de relatorio markdown para comparacao de modelos de recomendacao.

Inspirado no generate_markdown_report da Fase 1 (churn prediction),
adapta o conceito para sistemas de recomendacao com metricas Top-K,
declaracao de campeao e analise de trade-offs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from techchallenge_fase2.pipelines.model_result import (
    ModelResult,
    declare_champion,
    rank_models,
)

# Metricas canonicas para a tabela principal
CANONICAL_METRICS = ("precision", "recall", "ndcg", "map", "hit_rate")

# Cabecalhos em portugues para o relatorio
METRIC_LABELS: dict[str, str] = {
    "precision": "Precision",
    "recall": "Recall",
    "ndcg": "NDCG",
    "map": "MAP",
    "hit_rate": "Hit Rate",
}

ROLE_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "baseline_neural": "Baseline Neural",
    "champion_candidate": "Candidato a Campeao",
}


def _extract_k_values_from_metrics(
    metrics: dict[str, float],
) -> list[int]:
    """Extrai valores de K presentes nas chaves de metricas."""
    k_values: set[int] = set()
    for key in metrics:
        if "@" in key:
            _, k_str = key.rsplit("@", 1)
            try:
                k_values.add(int(k_str))
            except ValueError:
                continue
    return sorted(k_values)


def _get_k_values(results: list[ModelResult]) -> list[int]:
    """Extrai valores de K presentes nos resultados."""
    k_values: set[int] = set()
    for result in results:
        k_values.update(_extract_k_values_from_metrics(result.metrics))
    return sorted(k_values)


def build_comparison_table(
    results: list[ModelResult],
    k_values: tuple[int, ...] | None = None,
) -> str:
    """Constroi tabela markdown comparativa de modelos.

    Args:
        results: Lista de ModelResult.
        k_values: Valores de K para a tabela. Se None, detecta
            automaticamente.

    Returns:
        String com tabela markdown.
    """
    if not results:
        return "_Nenhum resultado disponivel._"

    if k_values is None:
        detected_ks = _get_k_values(results)
        k_values = tuple(detected_ks) if detected_ks else (10,)

    ranked = rank_models(results, k=k_values[0] if k_values else 10)

    # Cabecalho da tabela
    header_parts = ["Modelo", "Papel"]
    for k in k_values:
        for metric in CANONICAL_METRICS:
            label = METRIC_LABELS.get(metric, metric)
            header_parts.append(f"{label}@{k}")
    header_parts.extend(["H-Mean@10", "Treino (s)", "Infer. (s)"])

    # Linhas da tabela
    rows: list[str] = []
    for result in ranked:
        row_parts = [
            f"**{result.model_name}**",
            ROLE_LABELS.get(result.model_role, result.model_role),
        ]
        for k in k_values:
            for metric in CANONICAL_METRICS:
                val = result.metric_at_k(metric, k)
                row_parts.append(f"{val:.4f}")
        row_parts.append(f"{result.harmonic_mean_at_k(10):.4f}")
        row_parts.append(f"{result.train_time_sec:.2f}")
        row_parts.append(f"{result.infer_time_sec:.2f}")
        rows.append("| " + " | ".join(row_parts) + " |")

    header = "| " + " | ".join(header_parts) + " |"
    separator = "| " + " | ".join("---" for _ in header_parts) + " |"

    return "\n".join([header, separator] + rows)


def build_champion_section(
    results: list[ModelResult],
    k: int = 10,
) -> str:
    """Constroi secao markdown com declaracao do campeao.

    Args:
        results: Lista de ModelResult.
        k: Valor de K para comparacao.

    Returns:
        String com secao markdown do campeao.
    """
    champion, runner_up = declare_champion(results, k)

    if champion is None:
        return "### Campeao\n\n_Nenhum modelo avaliado._\n"

    role_label = ROLE_LABELS.get(champion.model_role, champion.model_role)
    hm = champion.harmonic_mean_at_k(k)
    lines = [
        "### Campeao",
        "",
        f"**{champion.model_name}** ({role_label}) com H-Mean@{k} = {hm:.4f}",
        "",
    ]

    if runner_up is not None:
        gain = champion.harmonic_mean_at_k(k) - runner_up.harmonic_mean_at_k(k)
        runner_up_hm = runner_up.harmonic_mean_at_k(k)
        if runner_up_hm > 0:
            rel_gain = gain / runner_up_hm * 100
            lines.append(
                f"Vantagem sobre **{runner_up.model_name}**: "
                f"+{gain:.4f} ({rel_gain:.1f}% relativo)"
            )
        else:
            lines.append(f"Vantagem sobre **{runner_up.model_name}**: +{gain:.4f}")
        lines.append("")

    # Metricas detalhadas do campeao
    lines.append("| Metrica | Valor |")
    lines.append("| --- | --- |")
    champion_k_values = _extract_k_values_from_metrics(champion.metrics)
    for metric in CANONICAL_METRICS:
        label = METRIC_LABELS.get(metric, metric)
        for k_val in champion_k_values:
            val = champion.metric_at_k(metric, k_val)
            lines.append(f"| {label}@{k_val} | {val:.4f} |")
    lines.append(f"| H-Mean@{k} | {hm:.4f} |")
    lines.append(f"| Tempo de treino | {champion.train_time_sec:.2f}s |")
    lines.append(f"| Tempo de inferencia | {champion.infer_time_sec:.2f}s |")

    return "\n".join(lines)


def build_tradeoff_section(
    results: list[ModelResult],
    k: int = 10,
) -> str:
    """Constroi secao markdown com analise de trade-offs entre modelos.

    Analisa a relacao entre precision e recall, e entre
    qualidade de recomendacao e tempo de inferencia.

    Args:
        results: Lista de ModelResult.
        k: Valor de K para a analise.

    Returns:
        String com secao markdown de trade-offs.
    """
    if not results:
        return ""

    ranked = rank_models(results, k)
    lines = [
        "### Trade-offs",
        "",
        "| Modelo | Precision@K | Recall@K | NDCG@K | Tempo Total (s) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for result in ranked:
        prec = result.metric_at_k("precision", k)
        rec = result.metric_at_k("recall", k)
        ndcg = result.metric_at_k("ndcg", k)
        total_time = result.train_time_sec + result.infer_time_sec
        lines.append(
            f"| {result.model_name} | {prec:.4f} | {rec:.4f} "
            f"| {ndcg:.4f} | {total_time:.2f} |"
        )

    lines.append("")
    lines.append(
        "_Precision favorece recomendacoes mais acuradas com menos "
        "itens; Recall favorece cobertura mais ampla dos itens "
        "relevantes. O trade-off entre qualidade e velocidade e "
        "importante para sistemas em producao._"
    )

    return "\n".join(lines)


def build_data_section(
    num_users: int = 0,
    num_items: int = 0,
    num_interactions: int = 0,
    num_evaluated_users: int = 0,
    dataset_name: str = "RetailRocket E-Commerce",
    split_strategy: str = "chronological_3way",
    test_ratio: float = 0.15,
    val_ratio: float | None = None,
    random_seed: int = 42,
) -> str:
    """Constroi secao markdown com resumo dos dados.

    Args:
        num_users: Numero total de usuarios.
        num_items: Numero total de itens.
        num_interactions: Numero total de interacoes.
        num_evaluated_users: Numero de usuarios avaliados.
        dataset_name: Nome do dataset.
        split_strategy: Estrategia de divisao.
        test_ratio: Fracao reservada para teste.
        val_ratio: Fracao reservada para validacao (3-way split).
            Se None, usa split 2-way.
        random_seed: Seed para reprodutibilidade.

    Returns:
        String com secao markdown de dados.
    """
    sparsity = (
        1.0 - (num_interactions / (num_users * num_items))
        if num_users and num_items
        else 1.0
    )
    if val_ratio is not None:
        train_pct = f"{1 - test_ratio - val_ratio:.0%}"
        val_pct = f"{val_ratio:.0%}"
        test_pct = f"{test_ratio:.0%}"
        split_desc = (
            f"{split_strategy} ({train_pct} treino / {val_pct} val / {test_pct} teste)"
        )
    else:
        train_pct = f"{1 - test_ratio:.0%}"
        test_pct = f"{test_ratio:.0%}"
        split_desc = f"{split_strategy} ({train_pct} treino / {test_pct} teste)"

    lines = [
        "### Dados",
        "",
        f"- **Dataset**: {dataset_name}",
        f"- **Estrategia de divisao**: {split_desc}",
        f"- **Usuarios totais**: {num_users:,}",
        f"- **Itens totais**: {num_items:,}",
        f"- **Interacoes totais**: {num_interactions:,}",
        f"- **Usuarios avaliados**: {num_evaluated_users:,}",
        f"- **Esparsidade**: {sparsity:.4%}",
        f"- **Seed**: {random_seed}",
    ]
    return "\n".join(lines)


def generate_markdown_report(  # noqa: PLR0913
    results: list[ModelResult],
    k_values: tuple[int, ...] = (5, 10, 20),
    champion_k: int = 10,
    num_users: int = 0,
    num_items: int = 0,
    num_interactions: int = 0,
    num_evaluated_users: int = 0,
    dataset_name: str = "RetailRocket E-Commerce",
    split_strategy: str = "chronological_3way",
    test_ratio: float = 0.15,
    val_ratio: float | None = 0.15,
    random_seed: int = 42,
) -> str:
    """Gera relatorio markdown completo de comparacao de modelos.

    Inspirado no generate_markdown_report da Fase 1, adaptado para
    sistemas de recomendacao com metricas Top-K.

    Args:
        results: Lista de ModelResult com resultados de cada modelo.
        k_values: Valores de K avaliados.
        champion_k: Valor de K para declaracao do campeao.
        num_users: Numero total de usuarios no dataset.
        num_items: Numero total de itens no dataset.
        num_interactions: Numero total de interacoes.
        num_evaluated_users: Numero de usuarios avaliados.
        dataset_name: Nome do dataset.
        split_strategy: Estrategia de divisao treino/val/teste.
        test_ratio: Fracao reservada para teste.
        val_ratio: Fracao reservada para validacao (3-way split).
        random_seed: Seed para reprodutibilidade.

    Returns:
        String com relatorio markdown completo.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ranked = rank_models(results, champion_k)
    k_str = ", ".join(str(k) for k in k_values)

    sections = [
        "# Relatorio de Comparacao de Modelos de Recomendacao",
        "",
        f"_Gerado em {now}_",
        "",
        "## Resumo",
        "",
        build_data_section(
            num_users=num_users,
            num_items=num_items,
            num_interactions=num_interactions,
            num_evaluated_users=num_evaluated_users,
            dataset_name=dataset_name,
            split_strategy=split_strategy,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed,
        ),
        "",
        "## Resultados",
        "",
        f"### Tabela Comparativa (metricas em K={k_str})",
        "",
        build_comparison_table(results, k_values=k_values),
        "",
        build_champion_section(results, k=champion_k),
        "",
        build_tradeoff_section(results, k=champion_k),
        "",
        "## Ranking Completo",
        "",
    ]

    for i, result in enumerate(ranked, 1):
        hm = result.harmonic_mean_at_k(champion_k)
        role_label = ROLE_LABELS.get(result.model_role, result.model_role)
        sections.append(
            f"{i}. **{result.model_name}** "
            f"({role_label}) "
            f"- H-Mean@{champion_k}: {hm:.4f}"
        )

    sections.extend(
        [
            "",
            "---",
            "",
            "_Relatorio gerado automaticamente pela pipeline de baselines._",
        ]
    )

    return "\n".join(sections)


def save_markdown_report(
    content: str,
    output_path: str | Path,
) -> Path:
    """Salva conteudo markdown em arquivo.

    Args:
        content: Conteudo markdown.
        output_path: Caminho do arquivo de saida.

    Returns:
        Caminho do arquivo salvo.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
