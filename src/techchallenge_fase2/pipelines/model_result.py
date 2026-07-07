"""Container estruturado para resultados de avaliacao de modelos.

Inspirado no ModelResult da Fase 1 (churn prediction), adapta o conceito
para sistemas de recomendacao, agrupando metricas, tempos e metadados
de cada modelo em um unico objeto imutavel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Resultado estruturado da avaliacao de um modelo de recomendacao.

    Agrupa todas as informacoes de um modelo avaliado: metricas em
    multiplos K, tempos de treino/inferencia, papel na comparacao
    e metadados adicionais.

    Attributes:
        model_name: Nome identificador do modelo (ex: "popularity").
        model_role: Papel na comparacao ("baseline", "baseline_neural",
            "champion_candidate").
        metrics: Dicionario com metricas no formato "metrica@K" -> valor.
            Exemplo: {"precision@10": 0.12, "recall@10": 0.05, ...}.
        train_time_sec: Tempo de treino em segundos.
        infer_time_sec: Tempo de inferencia em segundos.
        num_users_evaluated: Numero de usuarios avaliados.
        catalog_coverage: Fracao do catalogo coberta pelas recomendacoes.
        extra: Metadados adicionais (hiperparametros, configuracoes, etc.).
    """

    model_name: str
    model_role: str
    metrics: dict[str, float]
    train_time_sec: float = 0.0
    infer_time_sec: float = 0.0
    num_users_evaluated: int = 0
    catalog_coverage: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def metric_at_k(self, metric_name: str, k: int) -> float:
        """Retorna o valor de uma metrica em um K especifico.

        Args:
            metric_name: Nome da metrica (ex: "precision", "recall").
            k: Valor de K (ex: 5, 10, 20).

        Returns:
            Valor da metrica ou 0.0 se nao encontrada.
        """
        key = f"{metric_name}@{k}"
        return self.metrics.get(key, 0.0)

    def harmonic_mean_at_k(self, k: int) -> float:
        """Calcula a media harmonica das 4 metricas canonicas em K.

        As metricas canonicas sao: precision, recall, ndcg e map.

        Args:
            k: Valor de K para as metricas.

        Returns:
            Media harmonica. Retorna 0.0 se alguma metrica for zero.
        """
        values = [
            self.metric_at_k(m, k) for m in ("precision", "recall", "ndcg", "map")
        ]
        if any(v <= 0 for v in values):
            return 0.0
        return len(values) / sum(1.0 / v for v in values)

    def to_comparison_dict(self) -> dict[str, Any]:
        """Converte o resultado para dicionario plano para DataFrame.

        Returns:
            Dicionario com model_name, model_role, todas as metricas,
            tempos e metadados.
        """
        result: dict[str, Any] = {
            "model": self.model_name,
            "model_role": self.model_role,
        }
        result.update(self.metrics)
        result["train_time_sec"] = self.train_time_sec
        result["infer_time_sec"] = self.infer_time_sec
        result["num_users_evaluated"] = self.num_users_evaluated
        result["catalog_coverage"] = self.catalog_coverage
        result["harmonic_mean_at_10"] = self.harmonic_mean_at_k(10)
        return result


def compute_harmonic_mean_at_k(
    results: list[ModelResult], k: int = 10
) -> dict[str, float]:
    """Calcula a media harmonica para cada modelo em um dado K.

    Args:
        results: Lista de ModelResult.
        k: Valor de K para o calculo.

    Returns:
        Dicionario mapeando nome do modelo para media harmonica.
    """
    return {r.model_name: r.harmonic_mean_at_k(k) for r in results}


def rank_models(results: list[ModelResult], k: int = 10) -> list[ModelResult]:
    """Ordena modelos pela media harmonica em K (decrescente).

    Args:
        results: Lista de ModelResult.
        k: Valor de K para ranking.

    Returns:
        Lista ordenada de ModelResult, melhor primeiro.
    """
    return sorted(results, key=lambda r: r.harmonic_mean_at_k(k), reverse=True)


def declare_champion(
    results: list[ModelResult],
    k: int = 10,
) -> tuple[ModelResult | None, ModelResult | None]:
    """Declara o modelo campeao com base na media harmonica em K.

    O campeao e o modelo com maior media harmonica. Se houver
    apenas um modelo, o runner_up sera None.

    Args:
        results: Lista de ModelResult.
        k: Valor de K para comparacao.

    Returns:
        Tupla (campeao, segundo_colocado) com ModelResult ou None.
    """
    if not results:
        return None, None
    ranked = rank_models(results, k)
    champion = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return champion, runner_up
