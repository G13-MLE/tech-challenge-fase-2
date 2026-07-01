"""Model Card builder para artefatos MLflow de recomendação.

Segue o framework Model Cards for Model Reporting (Mitchell et al., 2019).
Gera dict JSON-serializável para ser logado via mlflow.log_dict().
"""

from __future__ import annotations

from typing import Any

_MODEL_INFO: dict[str, dict[str, str]] = {
    "popularity": {
        "model_name": "Popularity Recommender",
        "framework": "custom",
        "architecture": (
            "Ranking por frequência global de interações. "
            "Recomenda os itens mais populares independentemente "
            "do usuário."
        ),
    },
    "recent_items": {
        "model_name": "Recent Items Recommender",
        "framework": "custom",
        "architecture": (
            "Ranking por recência das interações. "
            "Recomenda os itens mais recentemente interagidos, "
            "independentemente do usuário."
        ),
    },
    "random": {
        "model_name": "Random Recommender",
        "framework": "custom",
        "architecture": (
            "Amostragem uniforme aleatória do catálogo de itens. "
            "Serve como lower bound: qualquer modelo personalizado "
            "deve superar o acaso para justificar sua complexidade."
        ),
    },
}

_METRIC_LABELS: dict[str, str] = {
    "precision@5": "Precision@5",
    "recall@5": "Recall@5",
    "ndcg@5": "NDCG@5",
    "map@5": "MAP@5",
    "hit_rate@5": "Hit Rate@5",
    "precision@10": "Precision@10",
    "recall@10": "Recall@10",
    "ndcg@10": "NDCG@10",
    "map@10": "MAP@10",
    "hit_rate@10": "Hit Rate@10",
    "precision@20": "Precision@20",
    "recall@20": "Recall@20",
    "ndcg@20": "NDCG@20",
    "map@20": "MAP@20",
    "hit_rate@20": "Hit Rate@20",
}

_INTENDED_USE = {
    "primary": (
        "Recomendar produtos para usuários de e-commerce "
        "com base em padrões de interação históricos."
    ),
    "users": [
        "Equipes de produto e data science",
        "Sistemas automatizados de recomendação",
    ],
    "out_of_scope": [
        "Recomendação personalizada em cold-start puro sem interações",
        "Aplicação direta em outros domínios sem revalidação",
        "Única base para decisões críticas de negócio",
    ],
}

_FACTORS = {
    "popularity_bias": {
        "description": (
            "Modelos baseados em popularidade tendem a recomendar "
            "itens já populares, perpetuando o viés de popularidade "
            "e limitando a descoberta de itens de cauda longa."
        ),
        "impact": (
            "Itens populares dominam as recomendações, itens nicho "
            "raramente aparecem mesmo quando relevantes."
        ),
    },
    "cold_start": {
        "description": (
            "Usuários e itens novos sem interações históricas não "
            "podem receber ou gerar recomendações personalizadas."
        ),
        "impact": (
            "Baselines não personalizados tratam todos os usuários "
            "de forma idêntica, ignorando preferências individuais."
        ),
    },
    "data_sparsity": {
        "description": (
            "A maioria dos usuários interage com poucos itens, "
            "resultando em matriz de interações esparsa."
        ),
        "impact": (
            "Métricas de recall tendem a ser baixas quando o "
            "conjunto de itens relevantes é grande."
        ),
    },
    "temporal_dynamics": {
        "description": (
            "Preferências de usuários e popularidade de itens mudam ao longo do tempo."
        ),
        "impact": (
            "Modelos treinados em dados históricos podem não "
            "capturar tendências recentes."
        ),
    },
}

_ETHICAL_CONSIDERATIONS = {
    "biases": [
        {
            "name": "Viés de popularidade",
            "description": (
                "Recomendar itens populares amplifica sua visibilidade, "
                "criando um ciclo de retroalimentação que marginaliza "
                "itens menos populares."
            ),
            "mitigation": (
                "Combinar com modelos baseados em conteúdo ou "
                "embedding; diversificar recomendações com "
                "estratégias como MMR."
            ),
        },
        {
            "name": "Filter bubble",
            "description": (
                "Recomendações baseadas apenas em interações passadas "
                "podem limitar a exposição do usuário a conteúdos novos "
                "ou diferentes."
            ),
            "mitigation": (
                "Incluir exploração aleatória (epsilon-greedy) e "
                "diversificação forçada nas recomendações."
            ),
        },
    ],
    "general_mitigations": [
        "Monitorar diversidade e cobertura das recomendações",
        "Avaliar métricas por segmento de usuário",
        "Combinar múltiplas estratégias de recomendação",
        "Auditar periodicamente vieses nos resultados",
    ],
}

_CAVEATS = {
    "limitations": [
        "Baselines não personalizam recomendações por usuário",
        "Dependência da qualidade e completude dos dados de interação",
        "Sem capacidade de capturar preferências latentes dos usuários",
        "Métricas offline podem não refletir satisfação real do usuário",
    ],
    "recommendations": [
        "Usar baselines como referência (lower bound) para modelos mais avançados",
        "Implementar A/B testing antes de deploy em produção",
        "Monitorar métricas de diversidade e cobertura além de acurácia",
        "Combinar com informações contextuais e demográficas quando disponível",
    ],
}


def build_model_card(  # noqa: PLR0912
    model_type: str, **values: str | float
) -> dict[str, Any]:
    """Constrói um Model Card dict para artefato MLflow.

    Args:
        model_type: Um de 'popularity', 'recent_items' ou nome customizado.
        **values: Valores do run para popular o card
            (ex: precision@5=0.4, num_users=1000, dataset_version="abc123").

    Returns:
        Dict com seções do Model Card, pronto para mlflow.log_dict().
    """
    info = _MODEL_INFO.get(
        model_type,
        {
            "model_name": model_type,
            "framework": "custom",
            "architecture": model_type,
        },
    )
    v: dict[str, Any] = dict(values)

    model_details = {
        "model_name": info["model_name"],
        "model_type": model_type,
        "framework": info["framework"],
        "architecture": info["architecture"],
        "version": v.get("model_version", "v1.0"),
        "seed": v.get("random_seed", 42),
        "dataset_version": v.get("dataset_version", "unknown"),
    }

    metric_entries: list[dict[str, Any]] = []
    for key, label in _METRIC_LABELS.items():
        if key in v:
            metric_entries.append({"metric": label, "key": key, "value": v[key]})

    metrics_section: dict[str, Any] = {
        "primary_metrics": metric_entries,
    }

    num_users = v.get("num_users", v.get("num_evaluated_users", 0))
    if num_users:
        metrics_section["num_evaluated_users"] = int(float(num_users))

    evaluation_data = {
        "dataset": "RetailRocket E-Commerce",
        "source": v.get("dataset_source", "data/raw/"),
        "split": v.get("split_strategy", "temporal_holdout"),
        "preprocessing": ("Filtragem de interações, divisão temporal treino/teste"),
    }

    return {
        "model_details": model_details,
        "intended_use": _INTENDED_USE,
        "factors": _FACTORS,
        "metrics": metrics_section,
        "evaluation_data": evaluation_data,
        "ethical_considerations": _ETHICAL_CONSIDERATIONS,
        "caveats_and_recommendations": _CAVEATS,
        "_framework": (
            "Model Cards for Model Reporting (Mitchell et al., ACM FAccT 2019)"
        ),
    }
