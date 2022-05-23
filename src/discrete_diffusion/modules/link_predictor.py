import math

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import Batch

from discrete_diffusion.utils import compute_self_similarities, get_graph_sizes_from_batch


class LinkPredictor(nn.Module):
    def __init__(self, node_embedder: DictConfig, feature_dim: int, time_dim: int):
        super().__init__()

        self.node_embedder = instantiate(node_embedder, feature_dim=feature_dim)

        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.GELU(), nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x: Batch, t: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :param t:

        :return: tensor (all_possible_edges_batch, )
        """
        graph_sizes = get_graph_sizes_from_batch(x)

        # (num_nodes_in_batch, embedding_dim)
        node_embeddings = self.node_embedder(x)
        time_embeddings = self.compute_time_embeddings(t, graph_sizes)
        assert node_embeddings.shape == time_embeddings.shape

        # (num_nodes_in_batch, embedding_dim)
        embeddings = torch.cat((node_embeddings, time_embeddings), dim=-1)

        # similarities = embeddings @ embeddings.T
        # similarities = torch.sigmoid(similarities)
        similarities = compute_self_similarities(embeddings)

        # mask = torch.block_diag(*[torch.ones(i, i) for i in graph_sizes]).type(torch.bool)
        mask = torch.block_diag(*[torch.triu(torch.ones(i, i), diagonal=1) for i in graph_sizes]).bool()
        flattened_adj_soft = similarities[mask]

        return flattened_adj_soft

    def compute_time_embeddings(self, timesteps: torch.Tensor, graph_sizes: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings and align them to node embeddings

        :param timesteps:
        :param graph_sizes:

        :return:
        """

        time_embeddings = self.time_embedder(timesteps)

        # (num_nodes_in_batch, embedding_dim)
        time_embeddings = torch.repeat_interleave(time_embeddings, graph_sizes, dim=0)

        return time_embeddings


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):

        super().__init__()

        self.dim = dim

    def forward(self, x):
        """

        :param x:
        :return:
        """
        device = x.device

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb
