"""EASE^ (Embarrassingly Shallow Autoencoder) em PyTorch.

Implementacao da solucao fechada normalizada proposta por Steck (2019):
    B = I - P / diag(P), onde P = (X^T X + lambda * I)^{-1}

A diagonal de B e zerada para impedir que o modelo "cole" o proprio item
como recomendacao, forcando o aprendizado de relacoes entre itens distintos.

Referencias:
    Steck, H. (2019). "Embarrassingly Shallow Autoencoders for Sparse Data".
        https://arxiv.org/abs/1905.03375
    Franck, J. "TorchEASE" (implementacao PyTorch de referencia).
        https://github.com/franckjay/TorchEASE
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from techchallenge_fase2.models.base import Interaction, RecommenderModel


@dataclass(frozen=True, slots=True)
class EASEConfig:
    """Configuracao do modelo EASE^.

    Args:
        lambda_reg: Termo de regularizacao L2 adicionado a diagonal de G.
            Valores eficazes tipicamente estao entre 1e2 e 1e4.
        max_items: Numero maximo de itens considerados no catalogo.
            Os mais populares sao mantidos para viabilizar a inversao em
            memoria. Use 0 para desativar o filtro (catalogo completo).
        batch_size: Tamanho do bloco de usuarios processados por vez na
            fase de predicao, para controlar o consumo de memoria.
        popularity_blending: Peso de popularidade adicionado ao score do
            modelo para suavizar recomendacoes de itens raros. Use 0.0
            para desativar o blending.
    """

    lambda_reg: float = 250.0
    max_items: int = 20000
    batch_size: int = 1000
    popularity_blending: float = 0.0

    def __post_init__(self) -> None:
        """Valida os hiperparametros de EASE."""
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg deve ser positivo")
        if self.max_items < 0:
            raise ValueError("max_items deve ser nao negativo")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        if self.popularity_blending < 0:
            raise ValueError("popularity_blending deve ser nao negativo")


class EASETorchRecommender(RecommenderModel):
    """Recomendador colaborativo item-item baseado em EASE^.

    Aprende uma matriz de pesos B (item-item) por solucao fechada,
    sem gradiente descendente, e gera predicoes via produto X @ B.
    """

    def __init__(self, config: EASEConfig | None = None) -> None:
        """Inicializa o recomendador com a configuracao fornecida.

        Args:
            config: Hiperparametros do EASE^. Usa defaults se None.
        """
        self._config = config or EASEConfig()
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_user: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._item_popularity: np.ndarray = np.array([], dtype=np.float64)
        self._item_log_popularity: np.ndarray = np.array([], dtype=np.float64)
        self._most_popular_items: list[int] = []
        self._seen_items: dict[int, set[int]] = {}
        self._b_matrix: torch.Tensor | None = None
        self._top_item_indices: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Treina o modelo aprendendo a matriz B de similaridades item-item.

        Args:
            interactions: Iteravel de pares (user_id, item_id) como strings.
        """
        materialized = list(interactions)
        self._build_mappings(materialized)
        self._build_seen_items(materialized)
        x_sparse = self._build_sparse_matrix(materialized)
        x_filtered, top_indices = self._filter_top_items(x_sparse)
        self._top_item_indices = top_indices
        self._item_popularity = np.asarray(x_sparse.sum(axis=0)).ravel()
        self._item_log_popularity = np.log1p(self._item_popularity)
        self._most_popular_items = self._compute_popular_ranking()
        gram = self._build_gram(x_filtered)
        p_inv = self._invert_gram(gram)
        self._b_matrix = self._build_b_matrix(p_inv)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Retorna os itens de maior pontuacao ainda nao consumidos.

        Args:
            user_id: Identificador do usuario como string.
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por relevancia.
        """
        if self._b_matrix is None:
            raise RuntimeError("Modelo precisa ser treinado antes de recomendar")
        recommendation_limit = limit if limit is not None else 10
        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            return self._recommend_cold_start(recommendation_limit)
        return self._recommend_warm_start(user_idx, recommendation_limit)

    # --- Construcao de mapeamentos e matrizes ---

    def _build_mappings(self, interactions: list[Interaction]) -> None:
        """Constroi mapeamentos bidirecionais str<->int para usuarios e itens."""
        user_ids = {uid for uid, _ in interactions}
        item_ids = {iid for _, iid in interactions}
        self._user_to_idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        self._idx_to_user = {idx: uid for uid, idx in self._user_to_idx.items()}
        self._item_to_idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        self._idx_to_item = {idx: iid for iid, idx in self._item_to_idx.items()}

    def _build_seen_items(self, interactions: list[Interaction]) -> None:
        """Registra o conjunto de itens ja consumidos por usuario."""
        self._seen_items.clear()
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                self._seen_items.setdefault(user_idx, set()).add(item_idx)

    def _build_sparse_matrix(self, interactions: list[Interaction]) -> sp.csr_matrix:
        """Constroi a matriz esparsa user-item com feedback implicito binario."""
        rows, cols = [], []
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx[user_id]
            item_idx = self._item_to_idx[item_id]
            rows.append(user_idx)
            cols.append(item_idx)
        data = np.ones(len(rows), dtype=np.float64)
        shape = (len(self._user_to_idx), len(self._item_to_idx))
        return sp.csr_matrix((data, (rows, cols)), shape=shape)

    def _filter_top_items(
        self, x_sparse: sp.csr_matrix
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Filtra o catalogo aos itens mais populares se max_items > 0."""
        n_items = x_sparse.shape[1]
        if self._config.max_items <= 0 or self._config.max_items >= n_items:
            return x_sparse, np.arange(n_items, dtype=np.int64)
        popularity = np.asarray(x_sparse.sum(axis=0)).ravel()
        top_indices = np.argsort(-popularity)[: self._config.max_items]
        return x_sparse[:, top_indices].astype(np.float64).tocsr(), top_indices

    def _build_gram(self, x_filtered: sp.csr_matrix) -> torch.Tensor:
        """Constroi a matriz de Gram G = X^T X + lambda * I em CPU (LAPACK)."""
        n_items = x_filtered.shape[1]
        gram_np = (x_filtered.T @ x_filtered).toarray()
        gram_np += self._config.lambda_reg * np.eye(n_items, dtype=gram_np.dtype)
        return torch.from_numpy(gram_np)

    def _invert_gram(self, gram: torch.Tensor) -> torch.Tensor:
        """Inverte a matriz de Gram usando backend CPU (LAPACK)."""
        return torch.linalg.inv(gram)

    def _build_b_matrix(self, p_inv: torch.Tensor) -> torch.Tensor:
        """Constroi B = I - P / diag(P) e zera a diagonal."""
        diag_p = torch.diagonal(p_inv)
        b_matrix = torch.eye(p_inv.shape[0]) - p_inv / diag_p.unsqueeze(0)
        b_matrix.fill_diagonal_(0.0)
        return b_matrix

    def _compute_popular_ranking(self) -> list[int]:
        """Retorna os indices dos itens ordenados por popularidade decrescente."""
        if self._item_popularity.size == 0:
            return []
        return np.argsort(-self._item_popularity).tolist()

    # --- Predicao ---

    def _recommend_warm_start(self, user_idx: int, limit: int) -> list[str]:
        """Gera recomendacoes para usuario com historico de interacoes."""
        b_matrix = self._b_matrix
        if b_matrix is None:
            return self._recommend_cold_start(limit)
        n_top = b_matrix.shape[0]
        scores = self._compute_user_scores(user_idx)
        self._exclude_seen_items(user_idx, scores)
        top_indices = torch.topk(scores, min(limit, n_top)).indices.tolist()
        return [self._idx_to_item[int(self._top_item_indices[i])] for i in top_indices]

    def _compute_user_scores(self, user_idx: int) -> torch.Tensor:
        """Calcula os scores do usuario via produto X_user @ B com blending."""
        b_matrix = self._b_matrix
        if b_matrix is None:
            raise RuntimeError("Modelo nao treinado")
        n_top = b_matrix.shape[0]
        user_row = np.zeros(n_top, dtype=np.float64)
        for item_idx in self._seen_items.get(user_idx, set()):
            pos = np.searchsorted(self._top_item_indices, item_idx)
            if pos < n_top and self._top_item_indices[pos] == item_idx:
                user_row[pos] = 1.0
        user_tensor = torch.from_numpy(user_row)
        scores = user_tensor @ self._b_matrix
        if self._config.popularity_blending > 0:
            pop = torch.from_numpy(self._item_log_popularity[self._top_item_indices])
            scores = scores + self._config.popularity_blending * pop
        return scores

    def _exclude_seen_items(self, user_idx: int, scores: torch.Tensor) -> None:
        """Atribui -infinito aos itens ja consumidos pelo usuario."""
        seen = self._seen_items.get(user_idx, set())
        for item_idx in seen:
            pos = np.searchsorted(self._top_item_indices, item_idx)
            top_arr = self._top_item_indices
            if pos < len(top_arr) and top_arr[pos] == item_idx:
                scores[pos] = float("-inf")

    def _recommend_cold_start(self, limit: int) -> list[str]:
        """Recomenda itens mais populares para usuarios sem historico."""
        recommended: list[str] = []
        for item_idx in self._most_popular_items[:limit]:
            recommended.append(self._idx_to_item[int(item_idx)])
        return recommended
