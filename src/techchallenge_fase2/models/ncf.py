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

    Mantem os pesos treinados e o conjunto de itens ja consumidos por
    usuario para que recomendacoes nao repitam interacoes anteriores.
    """

    def __init__(self, model: NeuralCollaborativeFiltering) -> None:
        """Inicializa o recomendador com um NCF ja treinado.

        Args:
            model: Modelo NCF com pesos definidos (treinados ou carregados).
        """
        self._model = model
        self._user_history: dict[int, set[int]] = {}

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Registra historico de itens consumidos por usuario.

        Args:
            interactions: Iteravel de pares (user_id, item_id) como strings.
        """
        for user_str, item_str in interactions:
            user_idx, item_idx = int(user_str), int(item_str)
            self._user_history.setdefault(user_idx, set()).add(item_idx)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Retorna os itens de maior pontuacao ainda nao consumidos.

        Args:
            user_id: Indice do usuario como string.
            limit: Numero maximo de itens a recomendar.

        Returns:
            Identificadores dos itens recomendados, ordenados por relevancia.
        """
        user_idx = int(user_id)
        scores = score_candidates(self._model, user_idx, self._model.config.num_items)
        for item_idx in self._user_history.get(user_idx, set()):
            scores[item_idx] = float("-inf")
        top_limit = limit if limit is not None else 10
        top_scores, top_items = torch.topk(scores, top_limit)
        recommendations: list[str] = []
        for position, item in enumerate(top_items.tolist()):
            if torch.isinf(top_scores[position]):
                continue
            recommendations.append(str(int(item)))
        return recommendations
