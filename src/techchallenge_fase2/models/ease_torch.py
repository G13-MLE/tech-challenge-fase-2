"""EASE^ (Embarrassingly Shallow Autoencoder) em PyTorch.

Implementacao da solucao fechada normalizada proposta por Steck (2019):
    B = I - P / diag(P), onde P = (X^T X + lambda * I)^{-1}

A diagonal de B e zerada para impedir que o modelo "cole" o próprio item
como recomendação, forcando o aprendizado de relacoes entre itens distintos.

Referencias:
    Steck, H. (2019). "Embarrassingly Shallow Autoencoders for Sparse Data".
        https://arxiv.org/abs/1905.03375
    Franck, J. "TorchEASE" (implementação PyTorch de referência).
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
            Valores eficazes tipicamente estão entre 1e2 e 1e4.
        max_items: Numero máximo de itens considerados no catalogo.
            Os mais populares são mantidos para viabilizar a inversão em
            memória. Use 0 para desativar o filtro (catalogo completo).
        batch_size: Tamanho do bloco de usuários processados por vez na
            fase de predição, para controlar o consumo de memória.
        popularity_blending: Peso de popularidade adicionado ao score do
            modelo para suavizar recomendações de itens raros. Use 0.0
            para desativar o blending.
        device: Dispositivo para computacao. "auto" detecta MPS se
            disponível e faz fallback para CPU. "cpu" força CPU.
            A inversão de matriz sempre usa CPU (LAPACK); o dispositivo
            e usado apenas para o produto matricial de predição em lote.
    """

    lambda_reg: float = 250.0
    max_items: int = 20000
    batch_size: int = 1000
    popularity_blending: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        """Valida os hiperparametros de EASE."""
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg deve ser positivo")
        if self.max_items < 0:
            raise ValueError("max_items deve ser não negativo")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        if self.popularity_blending < 0:
            raise ValueError("popularity_blending deve ser não negativo")
        if self.device not in ("auto", "cpu"):
            raise ValueError("device deve ser 'auto' ou 'cpu'")


def resolve_device(device: str) -> torch.device:
    """Resolve o dispositivo de computacao a partir da configuração.

    A inversão de matriz sempre usa CPU (LAPACK); o dispositivo retornado
    aqui e usado apenas para o produto matricial de predição em lote.

    Args:
        device: "auto" detecta MPS se disponível; "cpu" força CPU.

    Returns:
        torch.device configurado.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


class EASETorchRecommender(RecommenderModel):
    """Recomendador colaborativo item-item baseado em EASE^.

    Aprende uma matriz de pesos B (item-item) por solucao fechada,
    sem gradiente descendente, e gera predições via produto X @ B.
    """

    def __init__(self, config: EASEConfig | None = None) -> None:
        """Inicializa o recomendador com a configuração fornecida.

        Args:
            config: Hiperparametros do EASE^. Usa defaults se None.
        """
        self._config = config or EASEConfig()
        self._device = resolve_device(self._config.device)
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
        self.build_mappings(materialized)
        self.build_seen_items(materialized)
        x_sparse = self.build_sparse_matrix(materialized)
        x_filtered, top_indices = self.filter_top_items(x_sparse)
        self._top_item_indices = top_indices
        self._item_popularity = np.asarray(x_sparse.sum(axis=0)).ravel()
        self._item_log_popularity = np.log1p(self._item_popularity)
        self._most_popular_items = self.compute_popular_ranking()
        gram = self.build_gram(x_filtered)
        p_inv = self.invert_gram(gram)
        b_cpu = self.build_b_matrix(p_inv)
        self._b_matrix = self.move_b_to_device(b_cpu)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Retorna os itens de maior pontuação ainda não consumidos.

        Args:
            user_id: Identificador do usuário como string.
            limit: Numero máximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por relevância.
        """
        if self._b_matrix is None:
            raise RuntimeError("Modelo precisa ser treinado antes de recomendar")
        recommendation_limit = limit if limit is not None else 10
        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            return self.recommend_cold_start(recommendation_limit)
        return self.recommend_warm_start(user_idx, recommendation_limit)

    def recommend_batch(
        self, user_ids: list[str], limit: int | None = None
    ) -> dict[str, list[str]]:
        """Gera recomendações em lote processando usuários em blocos.

        Processa usuários em blocos de tamanho ``batch_size`` para controlar
        o consumo de memória, movendo apenas o lote atual para o dispositivo.

        Args:
            user_ids: Lista de identificadores de usuário como strings.
            limit: Numero máximo de itens a recomendar por usuário.

        Returns:
            Dicionario mapeando user_id -> lista de recomendações.
        """
        if self._b_matrix is None:
            raise RuntimeError("Modelo precisa ser treinado antes de recomendar")
        recommendation_limit = limit if limit is not None else 10
        results: dict[str, list[str]] = {}
        warm_users: list[tuple[str, int]] = []
        for uid in user_ids:
            idx = self._user_to_idx.get(uid)
            if idx is None:
                results[uid] = self.recommend_cold_start(recommendation_limit)
            else:
                warm_users.append((uid, idx))
        batch_size = self._config.batch_size
        for start in range(0, len(warm_users), batch_size):
            batch = warm_users[start : start + batch_size]
            batch_results = self.recommend_batch_warm(batch, recommendation_limit)
            results.update(batch_results)
        return results

    # --- Construcao de mapeamentos e matrizes ---

    def build_mappings(self, interactions: list[Interaction]) -> None:
        """Constrói mapeamentos bidirecionais str<->int para usuários e itens."""
        user_ids = {uid for uid, _ in interactions}
        item_ids = {iid for _, iid in interactions}
        self._user_to_idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        self._idx_to_user = {idx: uid for uid, idx in self._user_to_idx.items()}
        self._item_to_idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        self._idx_to_item = {idx: iid for iid, idx in self._item_to_idx.items()}

    def build_seen_items(self, interactions: list[Interaction]) -> None:
        """Registra o conjunto de itens já consumidos por usuário."""
        self._seen_items.clear()
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                self._seen_items.setdefault(user_idx, set()).add(item_idx)

    def build_sparse_matrix(self, interactions: list[Interaction]) -> sp.csr_matrix:
        """Constrói a matriz esparsa user-item com feedback implicito binario."""
        rows, cols = [], []
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx[user_id]
            item_idx = self._item_to_idx[item_id]
            rows.append(user_idx)
            cols.append(item_idx)
        data = np.ones(len(rows), dtype=np.float64)
        shape = (len(self._user_to_idx), len(self._item_to_idx))
        return sp.csr_matrix((data, (rows, cols)), shape=shape)

    def filter_top_items(
        self, x_sparse: sp.csr_matrix
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Filtra o catalogo aos itens mais populares se max_items > 0."""
        n_items = x_sparse.shape[1]
        if self._config.max_items <= 0 or self._config.max_items >= n_items:
            return x_sparse, np.arange(n_items, dtype=np.int64)
        popularity = np.asarray(x_sparse.sum(axis=0)).ravel()
        # Ordem ascendente e obrigatória: compute_user_scores,
        # exclude_seen_items e build_batch_matrix usam np.searchsorted,
        # que requer um array ordenado ascendentemente.
        top_indices = np.sort(np.argsort(-popularity)[: self._config.max_items])
        return x_sparse[:, top_indices].astype(np.float64).tocsr(), top_indices

    def build_gram(self, x_filtered: sp.csr_matrix) -> torch.Tensor:
        """Constrói a matriz de Gram G = X^T X + lambda * I em CPU (LAPACK)."""
        n_items = x_filtered.shape[1]
        gram_np = (x_filtered.T @ x_filtered).toarray()
        gram_np += self._config.lambda_reg * np.eye(n_items, dtype=gram_np.dtype)
        return torch.from_numpy(gram_np)

    def invert_gram(self, gram: torch.Tensor) -> torch.Tensor:
        """Inverte a matriz de Gram usando backend CPU (LAPACK)."""
        return torch.linalg.inv(gram)

    def build_b_matrix(self, p_inv: torch.Tensor) -> torch.Tensor:
        """Constrói B = I - P / diag(P) e zera a diagonal."""
        diag_p = torch.diagonal(p_inv)
        b_matrix = torch.eye(p_inv.shape[0]) - p_inv / diag_p.unsqueeze(0)
        b_matrix.fill_diagonal_(0.0)
        return b_matrix

    def move_b_to_device(self, b_cpu: torch.Tensor) -> torch.Tensor:
        """Move B para o dispositivo, convertendo para float32 se necessário.

        MPS e CUDA não suportam float64; a conversao para float32 e
        necessária nesses dispositivos. CPU mantem float64 para precisao.
        """
        if self._device.type == "cpu":
            return b_cpu.to(self._device)
        return b_cpu.float().to(self._device)

    def compute_popular_ranking(self) -> list[int]:
        """Retorna os índices dos itens ordenados por popularidade decrescente."""
        if self._item_popularity.size == 0:
            return []
        return np.argsort(-self._item_popularity).tolist()

    # --- Predicao ---

    def recommend_warm_start(self, user_idx: int, limit: int) -> list[str]:
        """Gera recomendações para usuário com histórico de interações."""
        b_matrix = self._b_matrix
        if b_matrix is None:
            return self.recommend_cold_start(limit)
        n_top = b_matrix.shape[0]
        scores = self.compute_user_scores(user_idx)
        self.exclude_seen_items(user_idx, scores)
        top_indices = torch.topk(scores, min(limit, n_top)).indices.tolist()
        return [self._idx_to_item[int(self._top_item_indices[i])] for i in top_indices]

    def recommend_batch_warm(
        self, batch: list[tuple[str, int]], limit: int
    ) -> dict[str, list[str]]:
        """Gera recomendações em lote para usuários com histórico.

        Constrói a matriz de interações do lote (X_batch) e computa
        X_batch @ B no dispositivo configurado, processando scores e
        exclusão de itens vistos em CPU para simplicidade.
        """
        b_matrix = self._b_matrix
        if b_matrix is None:
            return {uid: self.recommend_cold_start(limit) for uid, _ in batch}
        n_top = b_matrix.shape[0]
        x_batch = self.build_batch_matrix(batch, n_top)
        scores = self.compute_batch_scores(x_batch, b_matrix)
        return self.extract_batch_topk(batch, scores, limit, n_top)

    def build_batch_matrix(
        self, batch: list[tuple[str, int]], n_top: int
    ) -> torch.Tensor:
        """Constrói a matriz de interações do lote (users x top_items)."""
        x_batch = np.zeros((len(batch), n_top), dtype=np.float64)
        for row, (_, user_idx) in enumerate(batch):
            for item_idx in self._seen_items.get(user_idx, set()):
                pos = np.searchsorted(self._top_item_indices, item_idx)
                if pos < n_top and self._top_item_indices[pos] == item_idx:
                    x_batch[row, pos] = 1.0
        return torch.from_numpy(x_batch)

    def compute_batch_scores(
        self, x_batch: torch.Tensor, b_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Computa scores do lote via produto matricial no dispositivo."""
        x_dev = self.to_device_dtype(x_batch)
        scores = x_dev @ b_matrix
        if self._config.popularity_blending > 0:
            pop = self.popularity_tensor(b_matrix)
            scores = scores + self._config.popularity_blending * pop
        return scores.cpu()

    def extract_batch_topk(
        self,
        batch: list[tuple[str, int]],
        scores: torch.Tensor,
        limit: int,
        n_top: int,
    ) -> dict[str, list[str]]:
        """Exclui itens vistos e extrai top-K de cada linha de scores."""
        results: dict[str, list[str]] = {}
        for row, (uid, user_idx) in enumerate(batch):
            row_scores = scores[row].clone()
            self.exclude_seen_items(user_idx, row_scores)
            k = min(limit, n_top)
            top_indices = torch.topk(row_scores, k).indices.tolist()
            results[uid] = [
                self._idx_to_item[int(self._top_item_indices[i])] for i in top_indices
            ]
        return results

    def compute_user_scores(self, user_idx: int) -> torch.Tensor:
        """Calcula os scores do usuário via produto X_user @ B com blending."""
        b_matrix = self._b_matrix
        if b_matrix is None:
            raise RuntimeError("Modelo não treinado")
        n_top = b_matrix.shape[0]
        user_row = np.zeros(n_top, dtype=np.float64)
        for item_idx in self._seen_items.get(user_idx, set()):
            pos = np.searchsorted(self._top_item_indices, item_idx)
            if pos < n_top and self._top_item_indices[pos] == item_idx:
                user_row[pos] = 1.0
        user_tensor = self.to_device_dtype(torch.from_numpy(user_row))
        scores = user_tensor @ self._b_matrix
        if self._config.popularity_blending > 0:
            pop = self.popularity_tensor(self._b_matrix)
            scores = scores + self._config.popularity_blending * pop
        return scores.cpu()

    def to_device_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor para o dispositivo com dtype compatível.

        MPS e CUDA não suportam float64; converte para float32 antes
        de mover para o dispositivo. CPU mantem float64 para precisao.
        """
        if self._device.type == "cpu":
            return tensor.to(self._device)
        return tensor.float().to(self._device)

    def popularity_tensor(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Cria tensor de popularidade no dispositivo e dtype do ref_tensor."""
        pop_np = self._item_log_popularity[self._top_item_indices]
        pop = torch.from_numpy(pop_np)
        return self.to_device_dtype(pop)

    def exclude_seen_items(self, user_idx: int, scores: torch.Tensor) -> None:
        """Atribui -infinito aos itens já consumidos pelo usuário."""
        seen = self._seen_items.get(user_idx, set())
        for item_idx in seen:
            pos = np.searchsorted(self._top_item_indices, item_idx)
            top_arr = self._top_item_indices
            if pos < len(top_arr) and top_arr[pos] == item_idx:
                scores[pos] = float("-inf")

    def recommend_cold_start(self, limit: int) -> list[str]:
        """Recomenda itens mais populares para usuários sem histórico."""
        recommended: list[str] = []
        for item_idx in self._most_popular_items[:limit]:
            recommended.append(self._idx_to_item[int(item_idx)])
        return recommended
