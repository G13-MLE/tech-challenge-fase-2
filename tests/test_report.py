"""Testes unitarios para o módulo de relatório markdown."""

import tempfile
from pathlib import Path

from techchallenge_fase2.pipelines.model_result import ModelResult
from techchallenge_fase2.pipelines.report import (
    build_champion_section,
    build_comparison_table,
    build_data_section,
    build_tradeoff_section,
    generate_markdown_report,
    save_markdown_report,
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


class TestBuildComparisonTable:
    """Testes para build_comparison_table."""

    def test_empty_results_returns_message(self) -> None:
        """Tabela vazia retorna mensagem de nenhum resultado."""
        table = build_comparison_table([])
        assert "Nenhum resultado" in table

    def test_single_model_produces_table(self) -> None:
        """Tabela com um modelo contem o nome do modelo."""
        result = make_result(name="popularity")
        table = build_comparison_table([result], k_values=(10,))
        assert "popularity" in table

    def test_table_contains_metric_headers(self) -> None:
        """Tabela contem cabeçalhos das métricas."""
        result = make_result(
            name="ease_torch",
            role="champion_candidate",
        )
        table = build_comparison_table([result], k_values=(10,))
        assert "Precision@10" in table
        assert "Recall@10" in table
        assert "NDCG@10" in table
        assert "MAP@10" in table

    def test_table_ranks_by_harmonic_mean(self) -> None:
        """Modelos são ordenados por media harmonica decrescente."""
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
            recall=0.05,
            ndcg=0.08,
            map_val=0.06,
        )
        table = build_comparison_table([worst, best], k_values=(10,))
        # Melhor modelo aparece primeiro (apos cabeçalho e separador)
        lines = table.strip().split("\n")
        data_lines = [line for line in lines if "|" in line and "---" not in line]
        # data_lines[0] = cabeçalho, data_lines[1] = melhor modelo
        assert "best" in data_lines[1]


class TestBuildChampionSection:
    """Testes para build_champion_section."""

    def test_empty_results_returns_no_model(self) -> None:
        """Secao do campeão com resultados vazios indica nenhum modelo."""
        section = build_champion_section([])
        assert "Nenhum modelo" in section

    def test_declares_best_model_as_champion(self) -> None:
        """Campeão e o modelo com maior media harmonica."""
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
        section = build_champion_section([best, worst])
        assert "ease_torch" in section
        assert "Campeão" in section

    def test_shows_advantage_over_runner_up(self) -> None:
        """Secao mostra vantagem sobre o segundo colocado."""
        best = make_result(
            name="ease_torch",
            role="champion_candidate",
            precision=0.5,
            recall=0.5,
            ndcg=0.5,
            map_val=0.5,
        )
        worst = make_result(
            name="random",
            role="baseline",
            precision=0.1,
            recall=0.1,
            ndcg=0.1,
            map_val=0.1,
        )
        section = build_champion_section([best, worst])
        assert "Vantagem" in section


class TestBuildTradeoffSection:
    """Testes para build_tradeoff_section."""

    def test_empty_results_returns_empty_string(self) -> None:
        """Secao de trade-offs vazia retorna string vazia."""
        assert build_tradeoff_section([]) == ""

    def test_contains_model_names(self) -> None:
        """Secao de trade-offs contem nomes dos modelos."""
        result = make_result(name="popularity")
        section = build_tradeoff_section([result], k=10)
        assert "popularity" in section

    def test_contains_metric_headers(self) -> None:
        """Secao contem cabeçalhos de métricas."""
        result = make_result(name="popularity")
        section = build_tradeoff_section([result], k=10)
        assert "Precision" in section


class TestBuildDataSection:
    """Testes para build_data_section."""

    def test_contains_dataset_name(self) -> None:
        """Secao contem nome do dataset."""
        section = build_data_section(
            num_users=1000,
            num_items=500,
            dataset_name="RetailRocket",
        )
        assert "RetailRocket" in section

    def test_contains_user_and_item_counts(self) -> None:
        """Secao contem contagens de usuários e itens."""
        section = build_data_section(num_users=1000, num_items=500)
        assert "1,000" in section
        assert "500" in section

    def test_computes_sparsity(self) -> None:
        """Secao calcula esparsidade."""
        section = build_data_section(
            num_users=100,
            num_items=200,
            num_interactions=1000,
        )
        assert "Esparsidade" in section

    def test_3way_split_description_shows_validation(self) -> None:
        """Secao com val_ratio mostra split 3-way com validação."""
        section = build_data_section(
            num_users=100,
            num_items=200,
            num_interactions=1000,
            split_strategy="chronological_3way",
            test_ratio=0.15,
            val_ratio=0.15,
        )
        assert "val" in section.lower()
        assert "70% treino" in section
        assert "15% val" in section
        assert "15% teste" in section

    def test_2way_split_description_omits_validation(self) -> None:
        """Secao sem val_ratio mostra split 2-way sem validação."""
        section = build_data_section(
            num_users=100,
            num_items=200,
            num_interactions=1000,
            split_strategy="chronological_holdout",
            test_ratio=0.15,
            val_ratio=None,
        )
        assert "val" not in section.lower().split("estratégia")[1].split("(")[0]


class TestGenerateMarkdownReport:
    """Testes para generate_markdown_report."""

    def test_empty_results_produces_report(self) -> None:
        """Relatorio vazio e gerado sem erros."""
        report = generate_markdown_report(results=[])
        assert "Relatorio" in report

    def test_report_contains_model_names(self) -> None:
        """Relatorio contem nomes dos modelos."""
        results = [
            make_result(name="popularity"),
            make_result(name="random", precision=0.01),
        ]
        report = generate_markdown_report(results=results)
        assert "popularity" in report
        assert "random" in report

    def test_report_has_all_sections(self) -> None:
        """Relatorio contem todas as seções esperadas."""
        result = make_result(
            name="ease_torch",
            role="champion_candidate",
        )
        report = generate_markdown_report(results=[result])
        assert "Resumo" in report
        assert "Resultados" in report
        assert "Ranking" in report

    def test_report_includes_data_section(self) -> None:
        """Relatorio inclui seção de dados."""
        report = generate_markdown_report(
            results=[make_result()],
            num_users=1000,
            num_items=500,
            num_interactions=5000,
            num_evaluated_users=200,
        )
        assert "1,000" in report
        assert "500" in report


class TestSaveMarkdownReport:
    """Testes para save_markdown_report."""

    def test_creates_directory_and_file(self) -> None:
        """Salva relatório criando diretorios se necessário."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "report.md"
            result_path = save_markdown_report("# Test Report", path)
            assert result_path.exists()
            content = result_path.read_text(encoding="utf-8")
            assert content == "# Test Report"

    def test_overwrites_existing_file(self) -> None:
        """Sobrescreve arquivo existente."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text("old content", encoding="utf-8")
            save_markdown_report("new content", path)
            content = path.read_text(encoding="utf-8")
            assert content == "new content"
