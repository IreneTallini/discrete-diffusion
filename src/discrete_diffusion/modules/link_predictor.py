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

        self.node_embedder = instantiate(node_embedder, feature_dim=feature_dim, do_time_conditioning=True)

    def forward(self, x: Batch, t: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :param t:

        :return: tensor (all_possible_edges_batch, )
        """
        graph_sizes = get_graph_sizes_from_batch(x)

        # (num_nodes_in_batch, embedding_dim)
        node_embeddings = self.node_embedder(x, t)

        # (num_nodes_in_batch, embedding_dim)
        embeddings = node_embeddings

        similarities = embeddings @ embeddings.T

        m = similarities.min()
        M = similarities.max()
        similarities = (similarities - m) / (M - m + 0.0000001)

        mask = torch.block_diag(*[torch.triu(torch.ones(i, i), diagonal=1) for i in graph_sizes]).bool()
        flattened_adj_soft = similarities[mask]

        return flattened_adj_soft
