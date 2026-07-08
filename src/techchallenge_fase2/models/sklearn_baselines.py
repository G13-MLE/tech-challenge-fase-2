"""Baselines scikit-learn para recomendacao.

Implementa dois paradigmas de baseline:
- ItemKNNRecommender: vizinhanca por similaridade cosseno entre itens.
- LogisticRegressionRecommender: formulacao binaria user x item com
  sampling de negativos por usuario.

Ambos seguem o contrato `RecommenderModel` para integracao no Factory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from techchallenge_fase2.models.base import Interaction, RecommenderModel


@dataclass(frozen=True, slots=True)
class ItemKNNConfig:
    """Configuracao do recomendador ItemKNN.

    Args:
        n_neighbors: Numero de vizinhos mais proximos por item.
        metric: Metrica de similaridade passada ao NearestNeighbors.
    """

    n_neighbors: int = 20
    metric: str = "cosine"


class ItemKNNRecommender(RecommenderModel):
    """Recomendador por vizinhos mais proximos de itens (item-item KNN).

    Aprende similaridades entre itens a partir da co-ocorrencia em
    historicos de usuarios e recomenda itens similares aos consumidos.
    """

    def __init__(
        self,
        default_limit: int = 10,
        config: ItemKNNConfig | None = None,
    ) -> None:
        """Inicializa o recomendador ItemKNN.

        Args:
            default_limit: Numero padrao de itens retornados por recomendacao.
            config: Hiperparametros do KNN. Usa defaults se None.
        """
        self._default_limit = default_limit
        self._config = config or ItemKNNConfig()
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._seen_items: dict[int, set[int]] = {}
        self._item_popularity: np.ndarray = np.array([], dtype=np.float64)
        self._item_vectors: sp.csr_matrix | None = None
        self._knn: NearestNeighbors | None = None

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Treina o modelo ajustando o KNN sobre vetores de itens.

        Args:
            interactions: Iteravel de pares (user_id, item_id).
        """
        materialized = list(interactions)
        self.build_mappings(materialized)
        self.build_seen_items(materialized)
        self._item_vectors = self.build_sparse_matrix(materialized)
        self._item_popularity = np.asarray(self._item_vectors.sum(axis=0)).ravel()
        self._knn = self.fit_knn(self._item_vectors.T)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Recomenda itens agregando vizinhos dos itens ja consumidos.

        Args:
            user_id: Identificador do usuario como string.
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por score.
        """
        if self._knn is None or self._item_vectors is None:
            raise RuntimeError("Modelo precisa ser treinado antes de recomendar")
        recommendation_limit = limit if limit is not None else self._default_limit
        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            return self.recommend_cold_start(recommendation_limit)
        return self.recommend_warm_start(user_idx, recommendation_limit)

    def build_mappings(self, interactions: list[Interaction]) -> None:
        """Constroi mapeamentos str<->int para usuarios e itens."""
        user_ids = {uid for uid, _ in interactions}
        item_ids = {iid for _, iid in interactions}
        self._user_to_idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        self._item_to_idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        self._idx_to_item = {idx: iid for iid, idx in self._item_to_idx.items()}

    def build_seen_items(self, interactions: list[Interaction]) -> None:
        """Registra itens ja consumidos por usuario (indices internos int)."""
        self._seen_items.clear()
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                self._seen_items.setdefault(user_idx, set()).add(item_idx)

    def build_sparse_matrix(self, interactions: list[Interaction]) -> sp.csr_matrix:
        """Constroi a matriz esparsa user-item com feedback implicito binario."""
        rows, cols = [], []
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                rows.append(user_idx)
                cols.append(item_idx)
        data = np.ones(len(rows), dtype=np.float64)
        shape = (len(self._user_to_idx), len(self._idx_to_item))
        return sp.csr_matrix((data, (rows, cols)), shape=shape)

    def fit_knn(self, item_vectors: sp.csr_matrix) -> NearestNeighbors:
        """Ajusta o NearestNeighbors sobre os vetores de itens."""
        n_neighbors = min(self._config.n_neighbors, max(item_vectors.shape[0], 1))
        knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self._config.metric,
            algorithm="brute",
        )
        knn.fit(item_vectors)
        return knn

    def recommend_warm_start(self, user_idx: int, limit: int) -> list[str]:
        """Gera recomendacoes por agregacao de vizinhos dos itens consumidos."""
        scores = self.aggregate_scores(user_idx)
        self.exclude_seen_items(user_idx, scores)
        return self.top_items_from_scores(scores, limit)

    def aggregate_scores(self, user_idx: int) -> np.ndarray:
        """Soma similaridades dos vizinhos dos itens consumidos pelo usuario."""
        n_items = len(self._idx_to_item)
        scores = np.zeros(n_items, dtype=np.float64)
        if self._knn is None or self._item_vectors is None:
            return scores
        for item_idx in self._seen_items.get(user_idx, set()):
            _, indices = self._knn.kneighbors(self._item_vectors.T[item_idx])
            for neighbor_idx in indices.ravel():
                scores[neighbor_idx] += 1.0
        return scores

    def exclude_seen_items(self, user_idx: int, scores: np.ndarray) -> None:
        """Atribui -infinito aos itens ja consumidos pelo usuario."""
        for item_idx in self._seen_items.get(user_idx, set()):
            scores[item_idx] = float("-inf")

    def top_items_from_scores(self, scores: np.ndarray, limit: int) -> list[str]:
        """Retorna os top-L itens com maior score."""
        top_indices = np.argsort(-scores)[:limit]
        return [self._idx_to_item[int(idx)] for idx in top_indices]

    def recommend_cold_start(self, limit: int) -> list[str]:
        """Recomenda itens mais populares para usuarios sem historico."""
        if self._item_popularity.size == 0:
            return []
        top_indices = np.argsort(-self._item_popularity)[:limit]
        return [self._idx_to_item[int(idx)] for idx in top_indices]


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfig:
    """Configuracao do recomendador baseado em regressao logistica.

    Args:
        negatives_per_positive: Numero de negativos amostrados por positivo.
        C: Inverso da forca de regularizacao (valores menores = mais regularizacao).
        max_iter: Numero maximo de iteracoes do solver.
        random_seed: Seed para reprodutibilidade do sampling de negativos.
    """

    negatives_per_positive: int = 5
    C: float = 1.0
    max_iter: int = 100
    random_seed: int = 42


class LogisticRegressionRecommender(RecommenderModel):
    """Recomendador baseado em regressao logistica binaria user x item.

    Monta um problema de classificacao binaria com features one-hot de
    usuario e item concatenadas, treina um LogisticRegression para prever
    a probabilidade de interacao e pontua todos os itens para cada usuario.
    """

    def __init__(
        self,
        default_limit: int = 10,
        config: LogisticRegressionConfig | None = None,
    ) -> None:
        """Inicializa o recomendador por regressao logistica.

        Args:
            default_limit: Numero padrao de itens retornados por recomendacao.
            config: Hiperparametros da regressao. Usa defaults se None.
        """
        self._default_limit = default_limit
        self._config = config or LogisticRegressionConfig()
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_user: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._seen_items: dict[int, set[int]] = {}
        self._item_popularity: np.ndarray = np.array([], dtype=np.float64)
        self._model: LogisticRegression | None = None
        self._n_users = 0
        self._n_items = 0
        self._rng = np.random.default_rng(self._config.random_seed)

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Treina o LogisticRegression com features one-hot user+item.

        Args:
            interactions: Iteravel de pares (user_id, item_id).
        """
        materialized = list(interactions)
        self.build_mappings(materialized)
        self.build_seen_items(materialized)
        self._item_popularity = self.compute_item_popularity(materialized)
        x_train, y_train = self.build_training_set()
        self._model = self.train_logistic(x_train, y_train)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Recomenda itens por maior probabilidade predita ainda nao consumidos.

        Args:
            user_id: Identificador do usuario como string.
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por probabilidade.
        """
        if self._model is None:
            raise RuntimeError("Modelo precisa ser treinado antes de recomendar")
        recommendation_limit = limit if limit is not None else self._default_limit
        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            return self.recommend_cold_start(recommendation_limit)
        return self.recommend_warm_start(user_idx, recommendation_limit)

    def build_mappings(self, interactions: list[Interaction]) -> None:
        """Constroi mapeamentos str<->int para usuarios e itens."""
        user_ids = {uid for uid, _ in interactions}
        item_ids = {iid for _, iid in interactions}
        self._user_to_idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        self._idx_to_user = {idx: uid for uid, idx in self._user_to_idx.items()}
        self._item_to_idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        self._idx_to_item = {idx: iid for iid, idx in self._item_to_idx.items()}
        self._n_users = len(self._user_to_idx)
        self._n_items = len(self._item_to_idx)

    def build_seen_items(self, interactions: list[Interaction]) -> None:
        """Registra itens ja consumidos por usuario (indices internos int)."""
        self._seen_items.clear()
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                self._seen_items.setdefault(user_idx, set()).add(item_idx)

    def compute_item_popularity(self, interactions: list[Interaction]) -> np.ndarray:
        """Conta ocorrencias de cada item no conjunto de interacoes."""
        popularity = np.zeros(self._n_items, dtype=np.float64)
        for _, item_id in interactions:
            item_idx = self._item_to_idx.get(item_id)
            if item_idx is not None:
                popularity[item_idx] += 1.0
        return popularity

    def build_training_set(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Monta exemplos positivos e negativos com features one-hot user+item.

        Returns:
            Tupla (X, y) com features esparsas (n_users+n_items) e labels.
        """
        positives = self.collect_positives()
        negatives = self.sample_negatives(len(positives))
        all_pairs = np.concatenate([positives, negatives])
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        x_train = self.build_onehot_features(all_pairs)
        return x_train, labels

    def collect_positives(self) -> np.ndarray:
        """Coleta pares (user_idx, item_idx) das interacoes positivas."""
        rows, cols = [], []
        for user_idx, items in self._seen_items.items():
            for item_idx in items:
                rows.append(user_idx)
                cols.append(item_idx)
        if not rows:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(list(zip(rows, cols, strict=False)), dtype=np.int64)

    def sample_negatives(self, n_positives: int) -> np.ndarray:
        """Amostra negativos (user, item nao consumido) por positivo."""
        if n_positives == 0 or self._n_users == 0 or self._n_items == 0:
            return np.empty((0, 2), dtype=np.int64)
        n_negatives = n_positives * self._config.negatives_per_positive
        user_ids = self._rng.choice(self._n_users, size=n_negatives, replace=True)
        item_ids = self._rng.choice(self._n_items, size=n_negatives, replace=True)
        return np.array(list(zip(user_ids, item_ids, strict=False)), dtype=np.int64)

    def build_onehot_features(self, pairs: np.ndarray) -> sp.csr_matrix:
        """Constroi features esparsas concatenando one-hot de user e item.

        Args:
            pairs: Array (N, 2) com colunas [user_idx, item_idx].

        Returns:
            Matriz esparsa (N, n_users + n_items) com one-hot concatenado.
        """
        if len(pairs) == 0:
            return sp.csr_matrix((0, self._n_users + self._n_items))
        user_indices = pairs[:, 0]
        item_indices = pairs[:, 1] + self._n_users
        rows = np.concatenate([np.arange(len(pairs)), np.arange(len(pairs))])
        cols = np.concatenate([user_indices, item_indices])
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(pairs), self._n_users + self._n_items)
        )

    def train_logistic(
        self, x_train: sp.csr_matrix, y_train: np.ndarray
    ) -> LogisticRegression:
        """Treina o LogisticRegression com os exemplos montados."""
        model = LogisticRegression(
            C=self._config.C,
            max_iter=self._config.max_iter,
            random_state=self._config.random_seed,
            solver="liblinear",
        )
        model.fit(x_train, y_train)
        return model

    def recommend_warm_start(self, user_idx: int, limit: int) -> list[str]:
        """Pontua todos os itens para o usuario e retorna os top-L nao vistos."""
        if self._model is None:
            raise RuntimeError("Modelo nao treinado")
        x_candidates = self.build_candidate_features(user_idx)
        scores = self._model.decision_function(x_candidates)
        self.exclude_seen_items(user_idx, scores)
        return self.top_items_from_scores(scores, limit)

    def build_candidate_features(self, user_idx: int) -> sp.csr_matrix:
        """Constoi features one-hot user + item para todos os itens candidatos.

        A matriz resultante tem shape (n_items, n_users + n_items): cada
        linha i possui one-hot do usuario na primeira metade e one-hot do
        item i na segunda metade.
        """
        n_items = self._n_items
        user_rows = np.full(n_items, user_idx, dtype=np.int64)
        item_rows = np.arange(n_items, dtype=np.int64)
        user_part = sp.csr_matrix(
            (
                np.ones(n_items, dtype=np.float64),
                (np.arange(n_items), user_rows),
            ),
            shape=(n_items, self._n_users),
        )
        item_part = sp.csr_matrix(
            (
                np.ones(n_items, dtype=np.float64),
                (np.arange(n_items), item_rows),
            ),
            shape=(n_items, self._n_items),
        )
        return sp.hstack([user_part, item_part]).tocsr()

    def exclude_seen_items(self, user_idx: int, scores: np.ndarray) -> None:
        """Atribui -infinito aos itens ja consumidos pelo usuario."""
        for item_idx in self._seen_items.get(user_idx, set()):
            scores[item_idx] = float("-inf")

    def top_items_from_scores(self, scores: np.ndarray, limit: int) -> list[str]:
        """Retorna os top-L itens com maior score."""
        top_indices = np.argsort(-scores)[:limit]
        return [self._idx_to_item[int(idx)] for idx in top_indices]

    def recommend_cold_start(self, limit: int) -> list[str]:
        """Recomenda itens mais populares para usuarios sem historico."""
        if self._item_popularity.size == 0:
            return []
        top_indices = np.argsort(-self._item_popularity)[:limit]
        return [self._idx_to_item[int(idx)] for idx in top_indices]
