"""Early stopping para o loop de treinamento.

Monitora uma métrica de validação e interrompe o treino quando ela
deixa de melhorar por um número configuravel de épocas (patience),
preservando o estado do modelo de melhor desempenho.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    """Estado interno do EarlyStopping entre épocas.

    Args:
        best_score: Melhor pontuação de validação observada.
        waiting: Epocas consecutivas sem melhora.
        stopped: Indica se o treinamento deve parar.
    """

    best_score: float
    waiting: int
    stopped: bool


class EarlyStopping:
    """Interrompe o treino quando a métrica de validação para de melhorar.

    Args:
        patience: Epocas sem melhora antes de parar.
        min_delta: Melhoria mínima considerada significativa.
        mode: "min" para métricas decrescentes (loss) ou "max" para AUC/recall.
    """

    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"
    ) -> None:
        """Inicializa parâmetros e estado de acompanhamento.

        Args:
            patience: Numero de épocas sem melhora antes de parar o treino.
            min_delta: Variacao mínima para considerar melhora significativa.
            mode: "min" se menor e melhor, "max" se maior e melhor.
        """
        if patience < 1:
            raise ValueError("patience deve ser positivo")
        if mode not in {"min", "max"}:
            raise ValueError("mode deve ser 'min' ou 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._initial = float("inf") if mode == "min" else float("-inf")
        self.state = EarlyStoppingState(
            best_score=self._initial, waiting=0, stopped=False
        )

    def _is_improvement(self, score: float) -> bool:
        """Verifica se a nova pontuação melhora a melhor atual."""
        if self.mode == "min":
            return score < self.state.best_score - self.min_delta
        return score > self.state.best_score + self.min_delta

    def step(self, score: float) -> bool:
        """Atualiza o estado e indica se houve melhora.

        Args:
            score: Metrica de validação da época atual.

        Returns:
            True se este e o novo melhor estado (momento de salvar checkpoint).
        """
        if self._is_improvement(score):
            self.state.best_score = score
            self.state.waiting = 0
            return True
        self.state.waiting += 1
        if self.state.waiting >= self.patience:
            self.state.stopped = True
        return False

    @property
    def should_stop(self) -> bool:
        """Indica se o treinamento deve ser interrompido."""
        return self.state.stopped

    @property
    def best_score(self) -> float:
        """Retorna a melhor pontuação de validação observada."""
        return self.state.best_score
