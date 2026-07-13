"""Modelo neural de recomendacao (Neural Collaborative Filtering).

Combina um ramo GMF (Generalized Matrix Factorization) com um ramo MLP
(Multi-Layer Perceptron) sobre embeddings de usuarios e itens, fundindo
os dois sinais em uma unica probabilidade de interacao.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from techchallenge_fase2.models.base import Interaction, RecommenderModel


@dataclass(frozen=True, slots=True)
class NCFConfig:
    """Configuracao do modelo Neural Collaborative Filtering.

    Args:
        num_users: Total de usuarios unicos.
        num_items: Total de itens unicos.
        embedding_dim: Dimensao dos embeddings compartilhada por GMF e MLP.
        mlp_hidden_sizes: Tamanhos das camadas ocultas do ramo MLP.
        dropout: Probabilidade de dropout entre as camadas do MLP.
    """

    num_users: int
    num_items: int
    embedding_dim: int = 64
    mlp_hidden_sizes: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2


@dataclass(frozen=True, slots=True)
class NCFTrainingConfig:
    """Configuracao do treino inline do NCF para o pipeline de baselines.

    Args:
        epochs: Numero de epocas de treino.
        learning_rate: Taxa de aprendizado do Adam.
        negatives_per_positive: Negativos amostrados por positivo.
        batch_size: Tamanho do lote por passo de otimizacao.
        random_seed: Seed para reprodutibilidade.
    """

    epochs: int = 30
    learning_rate: float = 0.001
    negatives_per_positive: int = 4
    batch_size: int = 256
    random_seed: int = 42


def build_mlp_layers(
    input_size: int, hidden_sizes: tuple[int, ...], dropout: float
) -> nn.Sequential:
    """Constroi as camadas MLP com ReLU e dropout entre elas.

    Args:
        input_size: Dimensao da entrada (concatenacao de embeddings).
        hidden_sizes: Tamanhos sequenciais das camadas ocultas.
        dropout: Probabilidade de dropout aplicada apos cada ReLU.

    Returns:
        Sequential pronto para o ramo MLP do NCF.
    """
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(previous, size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        previous = size
    return nn.Sequential(*layers)


class NeuralCollaborativeFiltering(nn.Module):
    """Rede neural embedding-based para recomendacao (GMF + MLP)."""

    def __init__(self, config: NCFConfig) -> None:
        """Inicializa embeddings e cabecas dos ramos GMF e MLP.

        Args:
            config: Configuracao de hiperparametros e dimensoes.
        """
        super().__init__()
        self.config = config
        mlp_input = config.embedding_dim * 2
        self.user_gmf = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_gmf = nn.Embedding(config.num_items, config.embedding_dim)
        self.user_mlp = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_mlp = nn.Embedding(config.num_items, config.embedding_dim)
        self.mlp = build_mlp_layers(mlp_input, config.mlp_hidden_sizes, config.dropout)
        fusion_input = config.embedding_dim + config.mlp_hidden_sizes[-1]
        self.fusion = nn.Linear(fusion_input, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa embeddings com Xavier e camadas com Xavier uniform."""
        for embedding in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.fusion.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula logits de interacao para pares usuario-item.

        Args:
            user_idx: Tensores de indices de usuarios.
            item_idx: Tensores de indices de itens.

        Returns:
            Tensores de logits (antes da sigmoide) por par.
        """
        gmf_vector = self.user_gmf(user_idx) * self.item_gmf(item_idx)
        mlp_vector = torch.cat(
            [self.user_mlp(user_idx), self.item_mlp(item_idx)], dim=-1
        )
        mlp_output = self.mlp(mlp_vector)
        fused = torch.cat([gmf_vector, mlp_output], dim=-1)
        return self.fusion(fused).squeeze(-1)


def score_candidates(
    model: NeuralCollaborativeFiltering, user_idx: int, num_items: int
) -> torch.Tensor:
    """Pontua todos os itens candidatos para um usuario.

    Args:
        model: Modelo NCF treinado.
        user_idx: Indice do usuario alvo.
        num_items: Total de itens candidatos a pontuar.

    Returns:
        Tensores de logits para todos os itens do catalogo.
    """
    model.eval()
    with torch.no_grad():
        users = torch.full((num_items,), user_idx, dtype=torch.long)
        items = torch.arange(num_items, dtype=torch.long)
        return torch.sigmoid(model(users, items))


class NeuralRecommender(RecommenderModel):
    """Adaptador do NCF para o contrato de recomendacao do projeto.

    Suporta dois modos:
    - IDs inteiros: assume NCF ja treinado e so registra historico
      de itens vistos (compativel com pipeline DVC + checkpoint).
    - IDs string: mapeia str<->int, reconstrui o NCF com dimensoes
      corretas e treina inline com BCE loss + negativos amostrados
      (compativel com o pipeline de baselines comparativo).
    """

    def __init__(
        self,
        model: NeuralCollaborativeFiltering,
        training_config: NCFTrainingConfig | None = None,
    ) -> None:
        """Inicializa o recomendador com um NCF e config de treino opcional.

        Args:
            model: Modelo NCF com pesos definidos (treinados ou aleatorios).
            training_config: Config de treino inline. Se None, usa defaults.
        """
        self._model = model
        self._training_config = training_config or NCFTrainingConfig()
        self._user_history: dict[int, set[int]] = {}
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_user: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._was_trained_inline = False

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Registra historico e treina o NCF inline se IDs forem string.

        Args:
            interactions: Iteravel de pares (user_id, item_id).
        """
        materialized = list(interactions)
        if self.needs_string_mapping(materialized):
            self.fit_with_string_ids(materialized)
        else:
            self.fit_with_integer_ids(materialized)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Retorna os itens de maior pontuacao ainda nao consumidos.

        Args:
            user_id: Identificador do usuario (string ou inteiro como str).
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por relevancia.
        """
        user_idx = self.resolve_user_idx(user_id)
        if user_idx is None:
            return self.recommend_cold_start(limit)
        scores = score_candidates(self._model, user_idx, self._model.config.num_items)
        for item_idx in self._user_history.get(user_idx, set()):
            scores[item_idx] = float("-inf")
        top_limit = limit if limit is not None else 10
        top_scores, top_items = torch.topk(scores, top_limit)
        recommendations: list[str] = []
        for position, item in enumerate(top_items.tolist()):
            if torch.isinf(top_scores[position]):
                continue
            recommendations.append(self.format_item_id(int(item)))
        return recommendations

    # --- Treino inline para IDs string ---

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
        """Mapeia str->int, reconstrui NCF e treina inline."""
        self.build_string_mappings(interactions)
        numeric_interactions = self.to_numeric_interactions(interactions)
        self.rebuild_model_for_string_ids()
        self.train_inline(numeric_interactions)
        self.register_history(numeric_interactions)
        self._was_trained_inline = True

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
        """Reconstroi o NCF com dimensoes derivadas do mapeamento."""
        config = NCFConfig(
            num_users=len(self._user_to_idx),
            num_items=len(self._item_to_idx),
            embedding_dim=self._model.config.embedding_dim,
            mlp_hidden_sizes=self._model.config.mlp_hidden_sizes,
            dropout=self._model.config.dropout,
        )
        self._model = NeuralCollaborativeFiltering(config)

    def train_inline(self, numeric_interactions: list[tuple[int, int]]) -> None:
        """Treina o NCF com BCE loss e negativos amostrados por usuario."""
        cfg = self._training_config
        torch.manual_seed(cfg.random_seed)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1
        )
        criterion = nn.BCEWithLogitsLoss()
        self._model.train()
        positives = numeric_interactions
        negatives = self.sample_negatives(positives)
        dataset = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        for _epoch in range(cfg.epochs):
            self.run_training_epoch(
                optimizer, criterion, dataset, labels, cfg.batch_size
            )
            scheduler.step()

    def sample_negatives(
        self, positives: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Amostra negativos por usuario, excluindo itens do historico positivo.

        Para cada interacao positiva (u, i), amostra
        negatives_per_positive itens que o usuario u nao consumiu,
        garantindo que nao ha ruido de label nos negativos.
        """
        cfg = self._training_config
        n_items = self._model.config.num_items
        # Constroi historico positivo por usuario
        user_positives: dict[int, set[int]] = {}
        for user_idx, item_idx in positives:
            user_positives.setdefault(user_idx, set()).add(item_idx)
        gen = torch.Generator().manual_seed(cfg.random_seed)
        negatives: list[tuple[int, int]] = []
        for user_idx, item_idx in positives:
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
                negatives.append((user_idx, neg_item))
        return negatives

    def run_training_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        dataset: list[tuple[int, int]],
        labels: list[float],
        batch_size: int,
    ) -> None:
        """Executa uma epoca de treino em lotes."""
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]
            users = torch.tensor([u for u, _ in batch], dtype=torch.long)
            items = torch.tensor([i for _, i in batch], dtype=torch.long)
            targets = torch.tensor(batch_labels, dtype=torch.float32)
            optimizer.zero_grad()
            logits = self._model(users, items)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    # --- Modo IDs inteiros (compatibilidade com pipeline DVC) ---

    def fit_with_integer_ids(self, interactions: list[Interaction]) -> None:
        """Registra historico assumindo IDs ja codificados como inteiros."""
        for user_str, item_str in interactions:
            user_idx, item_idx = int(user_str), int(item_str)
            self._user_history.setdefault(user_idx, set()).add(item_idx)

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
