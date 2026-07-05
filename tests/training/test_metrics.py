"""Tests for recommendation metrics module."""

from __future__ import annotations

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


class TestPrecisionAtK:
    """Tests for precision_at_k."""

    @staticmethod
    def test_all_relevant() -> None:
        """Todos os itens recomendados são relevantes."""
        relevant = {"a", "b", "c"}
        recommended = ["a", "b", "c", "d", "e"]
        assert precision_at_k(relevant, recommended, 3) == 1.0

    @staticmethod
    def test_none_relevant() -> None:
        """Nenhum item recomendado é relevante."""
        relevant = {"x", "y", "z"}
        recommended = ["a", "b", "c", "d", "e"]
        assert precision_at_k(relevant, recommended, 3) == 0.0

    @staticmethod
    def test_partial_relevance() -> None:
        """Alguns itens relevantes nos K primeiros."""
        relevant = {"a", "c"}
        recommended = ["a", "b", "c", "d", "e"]
        # Top-3: [a, b, c] -> 2 relevantes / 3 = 0.6667
        assert abs(precision_at_k(relevant, recommended, 3) - 2 / 3) < 1e-6

    @staticmethod
    def test_empty_relevant() -> None:
        """Conjunto de relevantes vazio."""
        assert precision_at_k(set(), ["a", "b"], 5) == 0.0

    @staticmethod
    def test_empty_recommended() -> None:
        """Lista de recomendados vazia."""
        assert precision_at_k({"a"}, [], 5) == 0.0

    @staticmethod
    def test_k_zero() -> None:
        """K igual a zero."""
        assert precision_at_k({"a"}, ["a"], 0) == 0.0

    @staticmethod
    def test_k_larger_than_list() -> None:
        """K maior que a lista de recomendados."""
        relevant = {"a", "b"}
        recommended = ["a", "c"]
        # Top-5: [a, c] -> 1 relevante / 5... wait, k=5 but list has 2
        # precision_at_k returns hits/k, so 1/5 = 0.2
        assert precision_at_k(relevant, recommended, 5) == 0.2


class TestRecallAtK:
    """Tests for recall_at_k."""

    @staticmethod
    def test_all_relevant_found() -> None:
        """Todos os itens relevantes estao nos K primeiros."""
        relevant = {"a", "b"}
        recommended = ["a", "b", "c", "d", "e"]
        assert recall_at_k(relevant, recommended, 2) == 1.0

    @staticmethod
    def test_no_relevant_found() -> None:
        """Nenhum item relevante nos K primeiros."""
        relevant = {"x", "y"}
        recommended = ["a", "b", "c", "d", "e"]
        assert recall_at_k(relevant, recommended, 3) == 0.0

    @staticmethod
    def test_partial_recall() -> None:
        """Recall parcial."""
        relevant = {"a", "b", "c"}
        recommended = ["a", "x", "b", "y", "z"]
        # Top-3: [a, x, b] -> 2 relevantes de 3 = 0.6667
        assert abs(recall_at_k(relevant, recommended, 3) - 2 / 3) < 1e-6

    @staticmethod
    def test_empty_relevant() -> None:
        """Conjunto de relevantes vazio retorna 0."""
        assert recall_at_k(set(), ["a"], 5) == 0.0

    @staticmethod
    def test_k_zero() -> None:
        """K igual a zero."""
        assert recall_at_k({"a"}, ["a"], 0) == 0.0


class TestNDCGAtK:
    """Tests for ndcg_at_k."""

    @staticmethod
    def test_perfect_ranking() -> None:
        """Todos os itens relevantes nos top positions."""
        relevant = {"a", "b"}
        recommended = ["a", "b", "c", "d", "e"]
        # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309
        # IDCG = 1/log2(2) + 1/log2(3) = same
        assert abs(ndcg_at_k(relevant, recommended, 5) - 1.0) < 1e-6

    @staticmethod
    def test_no_relevant() -> None:
        """Nenhum item relevante."""
        relevant = {"x", "y"}
        recommended = ["a", "b", "c"]
        assert ndcg_at_k(relevant, recommended, 3) == 0.0

    @staticmethod
    def test_worst_ranking() -> None:
        """Itens relevantes nas últimas posições."""
        relevant = {"a", "b"}
        recommended = ["c", "d", "a", "b", "e"]
        # NDCG should be < 1.0
        result = ndcg_at_k(relevant, recommended, 5)
        assert 0.0 < result < 1.0

    @staticmethod
    def test_empty_relevant() -> None:
        """Conjunto de relevantes vazio."""
        assert ndcg_at_k(set(), ["a", "b"], 5) == 0.0


class TestAveragePrecisionAtK:
    """Tests for average_precision_at_k."""

    @staticmethod
    def test_perfect_ranking() -> None:
        """Todos os itens relevantes nas primeiras posições."""
        relevant = {"a", "b"}
        recommended = ["a", "b", "c", "d", "e"]
        # AP@2: P@1=1.0, P@2=1.0 -> AP = (1.0+1.0)/2 = 1.0
        result = average_precision_at_k(relevant, recommended, 5)
        assert abs(result - 1.0) < 1e-6

    @staticmethod
    def test_no_relevant() -> None:
        """Nenhum item relevante."""
        relevant = {"x"}
        recommended = ["a", "b", "c"]
        assert average_precision_at_k(relevant, recommended, 3) == 0.0

    @staticmethod
    def test_empty_relevant() -> None:
        """Conjunto de relevantes vazio."""
        assert average_precision_at_k(set(), ["a", "b"], 5) == 0.0


class TestHitRateAtK:
    """Tests for hit_rate_at_k."""

    @staticmethod
    def test_hit() -> None:
        """Pelo menos um item relevante nos K primeiros."""
        relevant = {"a"}
        recommended = ["x", "a", "y"]
        assert hit_rate_at_k(relevant, recommended, 3) == 1.0

    @staticmethod
    def test_no_hit() -> None:
        """Nenhum item relevante nos K primeiros."""
        relevant = {"z"}
        recommended = ["a", "b", "c"]
        assert hit_rate_at_k(relevant, recommended, 3) == 0.0

    @staticmethod
    def test_empty_relevant() -> None:
        """Conjunto de relevantes vazio."""
        assert hit_rate_at_k(set(), ["a", "b"], 5) == 0.0


class TestMeanMetrics:
    """Tests for aggregated mean metrics."""

    @staticmethod
    def test_mean_precision_at_k() -> None:
        """Média de precision@k por usuário."""
        all_relevant = {
            "u1": {"a", "b"},
            "u2": {"c"},
        }
        all_recommended = {
            "u1": ["a", "b", "x"],
            "u2": ["c", "d", "e"],
        }
        # u1: P@2 = 2/2 = 1.0
        # u2: P@2 = 1/2 = 0.5
        # mean = 0.75
        result = mean_precision_at_k(all_relevant, all_recommended, 2)
        assert abs(result - 0.75) < 1e-6

    @staticmethod
    def test_mean_recall_at_k() -> None:
        """Média de recall@k por usuário."""
        all_relevant = {
            "u1": {"a", "b"},
            "u2": {"c", "d"},
        }
        all_recommended = {
            "u1": ["a", "x", "x"],
            "u2": ["c", "d", "x"],
        }
        # u1: R@2 = 1/2 = 0.5
        # u2: R@2 = 2/2 = 1.0
        # mean = 0.75
        result = mean_recall_at_k(all_relevant, all_recommended, 2)
        assert abs(result - 0.75) < 1e-6

    @staticmethod
    def test_mean_ndcg_at_k() -> None:
        """Média de ndcg@k por usuário."""
        all_relevant = {"u1": {"a"}}
        all_recommended = {"u1": ["a", "b"]}
        result = mean_ndcg_at_k(all_relevant, all_recommended, 2)
        assert abs(result - 1.0) < 1e-6

    @staticmethod
    def test_mean_map_at_k() -> None:
        """Média de AP@k por usuário (MAP@K)."""
        all_relevant = {"u1": {"a"}}
        all_recommended = {"u1": ["a", "b"]}
        result = mean_average_precision_at_k(all_relevant, all_recommended, 2)
        assert abs(result - 1.0) < 1e-6

    @staticmethod
    def test_mean_hit_rate_at_k() -> None:
        """Média de hit_rate@k por usuário."""
        all_relevant = {
            "u1": {"a"},
            "u2": {"z"},
        }
        all_recommended = {
            "u1": ["a", "b"],
            "u2": ["x", "y"],
        }
        # u1: hit=1.0, u2: hit=0.0
        # mean = 0.5
        result = mean_hit_rate_at_k(all_relevant, all_recommended, 2)
        assert abs(result - 0.5) < 1e-6

    @staticmethod
    def test_empty_mapping() -> None:
        """Mapeamento vazio retorna 0.0."""
        assert mean_precision_at_k({}, {}, 5) == 0.0
        assert mean_recall_at_k({}, {}, 5) == 0.0
        assert mean_ndcg_at_k({}, {}, 5) == 0.0
        assert mean_average_precision_at_k({}, {}, 5) == 0.0
        assert mean_hit_rate_at_k({}, {}, 5) == 0.0

    @staticmethod
    def test_missing_user_in_recommended() -> None:
        """Usuário em relevantes mas não em recomendados é ignorado."""
        all_relevant = {"u1": {"a"}, "u2": {"b"}}
        all_recommended = {"u1": ["a"]}
        # Apenas u1 é considerado; u2 não está em recommended
        result = mean_precision_at_k(all_relevant, all_recommended, 1)
        assert abs(result - 1.0) < 1e-6


class TestComputeRecommenderMetrics:
    """Tests for compute_recommender_metrics."""

    @staticmethod
    def test_returns_all_metrics() -> None:
        """Retorna todas as métricas para cada K."""
        all_relevant = {"u1": {"a", "b"}, "u2": {"c"}}
        all_recommended = {"u1": ["a", "b", "x"], "u2": ["c", "d", "e"]}
        k_values = (5, 10)

        metrics = compute_recommender_metrics(all_relevant, all_recommended, k_values)

        # Deve ter métricas para cada K
        for k in k_values:
            assert f"precision@{k}" in metrics
            assert f"recall@{k}" in metrics
            assert f"ndcg@{k}" in metrics
            assert f"map@{k}" in metrics
            assert f"hit_rate@{k}" in metrics

        assert "num_users" in metrics
        assert metrics["num_users"] == 2.0

    @staticmethod
    def test_default_k_values() -> None:
        """Valores de K padrão são (5, 10, 20)."""
        all_relevant = {"u1": {"a"}}
        all_recommended = {"u1": ["a"]}
        metrics = compute_recommender_metrics(all_relevant, all_recommended)
        assert "precision@5" in metrics
        assert "precision@10" in metrics
        assert "precision@20" in metrics
