"""PyTorch embedding recommender models.

Implementa recomendador baseado em embeddings com treino BPR
(Bayesian Personalized Ranking) para aprender representacoes
de usuarios e itens via dot product + biases.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from techchallenge_fase2.models.base import Interaction, RecommenderModel
from techchallenge_fase2.models.baselines import resolve_limit, validate_limit


@dataclass(frozen=True, slots=True)
class EmbeddingTrainingConfig:
    """Configuracao do treino inline do TorchEmbedding.

    Args:
        epochs: Numero de epocas de treino.
        learning_rate: Taxa de aprendizado do Adam.
        negatives_per_positive: Negativos amostrados por positivo.
        batch_size: Tamanho do lote por passo de otimizacao.
        random_seed: Seed para reprodutibilidade.
    """

    epochs: int = 30
    learning_rate: float = 0.005
    negatives_per_positive: int = 4
    batch_size: int = 256
    random_seed: int = 42


class EmbeddingScoringModel(nn.Module):
    """Score user-item pairs with trainable embeddings."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """Initialize embedding tables.

        Args:
            num_users: Number of encoded users.
            num_items: Number of encoded items.
            embedding_dim: Embedding vector size.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa embeddings com distribuicao normal leve."""
        for emb in (self.user_embeddings, self.item_embeddings):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for bias in (self.user_bias, self.item_bias):
            nn.init.zeros_(bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for user-item pairs."""
        user_vectors = self.user_embeddings(user_ids)
        item_vectors = self.item_embeddings(item_ids)
        dot_scores = (user_vectors * item_vectors).sum(dim=1)
        user_bias = self.user_bias(user_ids).squeeze(dim=1)
        item_bias = self.item_bias(item_ids).squeeze(dim=1)
        return dot_scores + user_bias + item_bias


class TorchEmbeddingRecommender(RecommenderModel):
    """Recommendation model backed by a PyTorch embedding network.

    Treina embeddings com BPR loss (bayesian personalized ranking)
    usando amostragem de negativos por usuario, excluindo itens
    ja consumidos. Na inferencia, pontua todos os itens candidatos
    para cada usuario e retorna os de maior score.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        default_limit: int = 10,
        training_config: EmbeddingTrainingConfig | None = None,
    ) -> None:
        """Initialize the recommender.

        Args:
            num_users: Number of encoded users.
            num_items: Number of encoded items.
            embedding_dim: Embedding vector size.
            default_limit: Default number of recommendations.
            training_config: Config de treino inline. Se None, usa defaults.
        """
        self.network = EmbeddingScoringModel(num_users, num_items, embedding_dim)
        self._default_limit = validate_limit(default_limit)
        self._training_config = training_config or EmbeddingTrainingConfig()
        self._user_history: dict[int, set[int]] = {}
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_user: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._was_trained_inline = False

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Treina embeddings com BPR loss e registra historico.

        Args:
            interactions: Iteravel de pares (user_id, item_id).
        """
        materialized = list(interactions)
        if not materialized:
            return
        if self.needs_string_mapping(materialized):
            self.fit_with_string_ids(materialized)
        else:
            self.fit_with_integer_ids(materialized)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Retorna os itens de maior pontuacao ainda nao consumidos.

        Args:
            user_id: Identificador do usuario.
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por relevancia.
        """
        user_idx = self.resolve_user_idx(user_id)
        if user_idx is None:
            return self.recommend_cold_start(limit)
        self.network.eval()
        recommendation_limit = resolve_limit(self._default_limit, limit)
        with torch.no_grad():
            users = torch.full(
                (self.network.item_embeddings.num_embeddings,),
                user_idx,
                dtype=torch.long,
            )
            items = torch.arange(
                self.network.item_embeddings.num_embeddings, dtype=torch.long
            )
            scores = self.network(users, items)
        for item_idx in self._user_history.get(user_idx, set()):
            scores[item_idx] = float("-inf")
        top_scores, top_items = torch.topk(scores, recommendation_limit)
        recommendations: list[str] = []
        for position, item in enumerate(top_items.tolist()):
            if torch.isinf(top_scores[position]):
                continue
            recommendations.append(self.format_item_id(int(item)))
        return recommendations

    # --- Treino com BPR ---

    def needs_string_mapping(self, interactions: list[Interaction]) -> bool:
        """Detecta se os IDs precisam mapeamento str->int."""
        if not interactions:
            return False
        sample_user, sample_item = interactions[0]
        try:
            int(sample_user)
            int(sample_item)
            return False
        except ValueError:
            return True

    def fit_with_string_ids(self, interactions: list[Interaction]) -> None:
        """Mapeia str->int, reconstrui rede e treina inline com BPR."""
        self.build_string_mappings(interactions)
        numeric_interactions = self.to_numeric_interactions(interactions)
        self.rebuild_model_for_string_ids()
        self.train_inline(numeric_interactions)
        self.register_history(numeric_interactions)
        self._was_trained_inline = True

    def fit_with_integer_ids(self, interactions: list[Interaction]) -> None:
        """Registra historico assumindo IDs ja codificados como inteiros."""
        for user_str, item_str in interactions:
            user_idx, item_idx = int(user_str), int(item_str)
            self._user_history.setdefault(user_idx, set()).add(item_idx)

    def build_string_mappings(self, interactions: list[Interaction]) -> None:
        """Constroi mapeamentos bidirecionais str<->int."""
        user_ids = {uid for uid, _ in interactions}
        item_ids = {iid for _, iid in interactions}
        self._user_to_idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        self._idx_to_user = {idx: uid for uid, idx in self._user_to_idx.items()}
        self._item_to_idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        self._idx_to_item = {idx: iid for iid, idx in self._item_to_idx.items()}

    def to_numeric_interactions(
        self, interactions: list[Interaction]
    ) -> list[tuple[int, int]]:
        """Converte interacoes string para pares (user_idx, item_idx)."""
        numeric: list[tuple[int, int]] = []
        for user_id, item_id in interactions:
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            if user_idx is not None and item_idx is not None:
                numeric.append((user_idx, item_idx))
        return numeric

    def rebuild_model_for_string_ids(self) -> None:
        """Reconstroi a rede com dimensoes derivadas do mapeamento."""
        num_users = len(self._user_to_idx)
        num_items = len(self._item_to_idx)
        embedding_dim = self.network.user_embeddings.embedding_dim
        self.network = EmbeddingScoringModel(num_users, num_items, embedding_dim)

    def train_inline(self, numeric_interactions: list[tuple[int, int]]) -> None:
        """Treina embeddings com BPR loss e negativos amostrados por usuario."""
        cfg = self._training_config
        torch.manual_seed(cfg.random_seed)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1
        )
        self.network.train()

        # Constroi historico positivo por usuario para amostragem
        user_positives: dict[int, set[int]] = {}
        for user_idx, item_idx in numeric_interactions:
            user_positives.setdefault(user_idx, set()).add(item_idx)

        gen = torch.Generator().manual_seed(cfg.random_seed)
        n_items = self.network.item_embeddings.num_embeddings

        for _epoch in range(cfg.epochs):
            # Gera tripletos BPR: (usuario, item_positivo, item_negativo)
            triplets = self.sample_bpr_triplets(
                numeric_interactions, user_positives, n_items, cfg, gen
            )
            self.run_bpr_epoch(optimizer, triplets, cfg.batch_size)
            scheduler.step()

    def sample_bpr_triplets(
        self,
        positives: list[tuple[int, int]],
        user_positives: dict[int, set[int]],
        n_items: int,
        cfg: EmbeddingTrainingConfig,
        gen: torch.Generator,
    ) -> list[tuple[int, int, int]]:
        """Amostra tripletos BPR (user, pos_item, neg_item) por interacao positiva."""
        triplets: list[tuple[int, int, int]] = []
        for user_idx, pos_item in positives:
            positive_items = user_positives.get(user_idx, set())
            neg_items: list[int] = []
            max_attempts = cfg.negatives_per_positive * 10
            attempts = 0
            while (
                len(neg_items) < cfg.negatives_per_positive and attempts < max_attempts
            ):
                candidate = torch.randint(0, n_items, (1,), generator=gen).item()
                attempts += 1
                if candidate not in positive_items:
                    neg_items.append(candidate)
            for neg_item in neg_items:
                triplets.append((user_idx, pos_item, neg_item))
        return triplets

    def run_bpr_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        triplets: list[tuple[int, int, int]],
        batch_size: int,
    ) -> None:
        """Executa uma epoca de treino BPR em lotes."""
        for start in range(0, len(triplets), batch_size):
            batch = triplets[start : start + batch_size]
            users = torch.tensor([u for u, _, _ in batch], dtype=torch.long)
            pos_items = torch.tensor([p for _, p, _ in batch], dtype=torch.long)
            neg_items = torch.tensor([n for _, _, n in batch], dtype=torch.long)

            pos_scores = self.network(users, pos_items)
            neg_scores = self.network(users, neg_items)

            # BPR loss: -log(sigmoid(pos_score - neg_score))
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Predicao ---

    def resolve_user_idx(self, user_id: str) -> int | None:
        """Converte user_id para indice interno, ou None se desconhecido."""
        if self._was_trained_inline:
            return self._user_to_idx.get(user_id)
        try:
            return int(user_id)
        except ValueError:
            return None

    def format_item_id(self, item_idx: int) -> str:
        """Converte indice interno do item de volta para string."""
        if self._was_trained_inline:
            return self._idx_to_item.get(item_idx, str(item_idx))
        return str(item_idx)

    def recommend_cold_start(self, limit: int | None) -> list[str]:
        """Recomendacao para usuario desconhecido: retorna lista vazia."""
        _ = limit
        return []

    def register_history(self, numeric_interactions: list[tuple[int, int]]) -> None:
        """Registra itens ja consumidos por usuario (indices internos)."""
        for user_idx, item_idx in numeric_interactions:
            self._user_history.setdefault(user_idx, set()).add(item_idx)
