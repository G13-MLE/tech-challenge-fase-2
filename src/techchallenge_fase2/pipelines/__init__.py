"""Pipeline de execucao, orquestracao de treinamento e estagios DVC."""

from techchallenge_fase2.pipelines.model_result import (
    ModelResult,
    compute_harmonic_mean_at_k,
    declare_champion,
    rank_models,
)
from techchallenge_fase2.pipelines.report import (
    build_champion_section,
    build_comparison_table,
    build_data_section,
    build_tradeoff_section,
    generate_markdown_report,
    save_markdown_report,
)

__all__ = [
    "ModelResult",
    "build_champion_section",
    "build_comparison_table",
    "build_data_section",
    "build_tradeoff_section",
    "compute_harmonic_mean_at_k",
    "declare_champion",
    "generate_markdown_report",
    "rank_models",
    "save_markdown_report",
]
