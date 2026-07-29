"""Testes unitarios para ModelResult."""

from techchallenge_fase2.pipelines.model_result import (
    ModelResult,
    compute_harmonic_mean_at_k,
    declare_champion,
    rank_models,
)


def make_result(
    name: str = "popularity",
    role: str = "baseline",
    precision: float = 0.1,
    recall: float = 0.05,
    ndcg: float = 0.08,
    map_val: float = 0.06,
    hit_rate: float = 0.2,
    k: int = 10,
) -> ModelResult:
    """Cria um ModelResult com métricas padrão para testes."""
    return ModelResult(
        model_name=name,
        model_role=role,
        metrics={
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"ndcg@{k}": ndcg,
            f"map@{k}": map_val,
            f"hit_rate@{k}": hit_rate,
            "num_users": 100.0,
        },
        train_time_sec=1.0,
        infer_time_sec=0.5,
        num_users_evaluated=100,
    )


class TestModelResult:
    """Testes para a dataclass ModelResult."""

    def test_metric_at_k_returns_correct_value(self) -> None:
        """metric_at_k retorna o valor da métrica no K especificado."""
        result = make_result(precision=0.15, k=10)
        assert result.metric_at_k("precision", 10) == 0.15

    def test_metric_at_k_returns_zero_for_missing_key(self) -> None:
        """metric_at_k retorna 0.0 para chave inexistente."""
        result = make_result(k=10)
        assert result.metric_at_k("precision", 5) == 0.0

    def test_harmonic_mean_at_k_computes_correctly(self) -> None:
        """harmonic_mean_at_k calcula a media harmonica das 4 métricas."""
        result = ModelResult(
            model_name="test",
            model_role="baseline",
            metrics={
                "precision@10": 0.25,
                "recall@10": 0.25,
                "ndcg@10": 0.25,
                "map@10": 0.25,
            },
        )
        # Todas iguais: H = 4 / (4 * 1/0.25) = 4 / 16 = 0.25
        assert result.harmonic_mean_at_k(10) == 0.25

    def test_harmonic_mean_at_k_returns_zero_if_any_is_zero(
        self,
    ) -> None:
        """harmonic_mean_at_k retorna 0.0 se alguma métrica for zero."""
        result = ModelResult(
            model_name="test",
            model_role="baseline",
            metrics={
                "precision@10": 0.1,
                "recall@10": 0.0,
                "ndcg@10": 0.08,
                "map@10": 0.06,
            },
        )
        assert result.harmonic_mean_at_k(10) == 0.0

    def test_harmonic_mean_at_k_with_different_values(self) -> None:
        """harmonic_mean_at_k calcula corretamente com valores distintos."""
        result = ModelResult(
            model_name="test",
            model_role="baseline",
            metrics={
                "precision@10": 0.5,
                "recall@10": 0.25,
                "ndcg@10": 0.4,
                "map@10": 0.2,
            },
        )
        expected = 4 / (1 / 0.5 + 1 / 0.25 + 1 / 0.4 + 1 / 0.2)
        assert abs(result.harmonic_mean_at_k(10) - expected) < 1e-10

    def test_to_comparison_dict_includes_all_fields(self) -> None:
        """to_comparison_dict inclui model, métricas, tempos e h-mean."""
        result = make_result(name="popularity", k=10)
        d = result.to_comparison_dict()
        assert d["model"] == "popularity"
        assert d["model_role"] == "baseline"
        assert "precision@10" in d
        assert "train_time_sec" in d
        assert "infer_time_sec" in d
        assert "harmonic_mean_at_10" in d
        assert "num_users_evaluated" in d

    def test_frozen_dataclass_is_immutable(self) -> None:
        """ModelResult e imutavel (frozen=True)."""
        result = make_result()
        try:
            result.model_name = "changed"  # type: ignore[misc]
            raise AssertionError("Deveria ter lancado FrozenInstanceError")
        except AttributeError:
            pass  # Esperado: dataclass frozen não permite atribuição


class TestRankModels:
    """Testes para rank_models."""

    def test_rank_models_orders_by_harmonic_mean_descending(
        self,
    ) -> None:
        """rank_models ordena por media harmonica decrescente."""
        best = make_result(
            name="best",
            precision=0.5,
            recall=0.5,
            ndcg=0.5,
            map_val=0.5,
        )
        worst = make_result(
            name="worst",
            precision=0.1,
            recall=0.1,
            ndcg=0.1,
            map_val=0.1,
        )
        mid = make_result(
            name="mid",
            precision=0.3,
            recall=0.3,
            ndcg=0.3,
            map_val=0.3,
        )

        ranked = rank_models([worst, best, mid], k=10)
        assert ranked[0].model_name == "best"
        assert ranked[1].model_name == "mid"
        assert ranked[2].model_name == "worst"

    def test_rank_models_empty_list(self) -> None:
        """rank_models retorna lista vazia para entrada vazia."""
        assert rank_models([], k=10) == []


class TestDeclareChampion:
    """Testes para declare_champion."""

    def test_declare_champion_returns_best_and_runner_up(
        self,
    ) -> None:
        """declare_champion retorna campeão e segundo colocado."""
        best = make_result(
            name="ease_torch",
            role="champion_candidate",
            precision=0.5,
        )
        worst = make_result(
            name="random",
            role="baseline",
            precision=0.01,
        )

        champion, runner_up = declare_champion([best, worst], k=10)
        assert champion is not None
        assert champion.model_name == "ease_torch"
        assert runner_up is not None
        assert runner_up.model_name == "random"

    def test_declare_champion_returns_none_for_empty_list(
        self,
    ) -> None:
        """declare_champion retorna (None, None) para lista vazia."""
        champion, runner_up = declare_champion([], k=10)
        assert champion is None
        assert runner_up is None

    def test_declare_champion_single_model(self) -> None:
        """declare_champion com único modelo retorna runner_up None."""
        result = make_result(name="popularity")
        champion, runner_up = declare_champion([result], k=10)
        assert champion is not None
        assert champion.model_name == "popularity"
        assert runner_up is None


class TestComputeHarmonicMeanAtK:
    """Testes para compute_harmonic_mean_at_k."""

    def test_computes_for_each_model(self) -> None:
        """compute_harmonic_mean_at_k calcula para cada modelo."""
        r1 = make_result(
            name="a",
            precision=0.25,
            recall=0.25,
            ndcg=0.25,
            map_val=0.25,
        )
        r2 = make_result(
            name="b",
            precision=0.5,
            recall=0.5,
            ndcg=0.5,
            map_val=0.5,
        )

        result = compute_harmonic_mean_at_k([r1, r2], k=10)
        assert result["a"] == 0.25
        assert result["b"] == 0.5

    def test_empty_list_returns_empty_dict(self) -> None:
        """compute_harmonic_mean_at_k retorna dict vazio para lista vazia."""
        assert compute_harmonic_mean_at_k([], k=10) == {}
