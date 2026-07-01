"""Módulo de treinamento e tracking MLflow para recomendação."""

from techchallenge_fase2.training.metrics import (
    average_precision_at_k,
    compute_recommender_metrics,
    hit_rate_at_k,
    mean_average_precision_at_k,
    mean_hit_rate_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_recall_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from techchallenge_fase2.training.mlflow_tracking import (
    MLflowConfig,
    log_artifacts,
    log_hyperparameters,
    log_input_data_summary,
    log_metrics,
    log_recommender_model,
    log_system_info,
    setup_mlflow,
)
from techchallenge_fase2.training.model_card import build_model_card
from techchallenge_fase2.training.plots import (
    save_catalog_coverage_plot,
    save_item_popularity_distribution,
    save_metrics_bar_chart,
    save_model_comparison_chart,
    save_precision_recall_at_k_curve,
)

__all__ = [
    "MLflowConfig",
    "average_precision_at_k",
    "build_model_card",
    "compute_recommender_metrics",
    "hit_rate_at_k",
    "log_artifacts",
    "log_hyperparameters",
    "log_input_data_summary",
    "log_metrics",
    "log_recommender_model",
    "log_system_info",
    "mean_average_precision_at_k",
    "mean_hit_rate_at_k",
    "mean_ndcg_at_k",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "save_catalog_coverage_plot",
    "save_item_popularity_distribution",
    "save_metrics_bar_chart",
    "save_model_comparison_chart",
    "save_precision_recall_at_k_curve",
    "setup_mlflow",
]
