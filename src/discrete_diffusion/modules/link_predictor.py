import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, GraphNorm, JumpingKnowledge

from discrete_diffusion.modules.mlp import MLP


class LinkPredictor(nn.Module):
    def __init__(self, node_embedder: DictConfig, feature_dim, time_dim):
        super().__init__()
        self.node_embedder = instantiate(node_embedder, feature_dim=feature_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.GELU(), nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x: Batch, t: torch.Tensor):
        node_embeddings = self.node_embedder(x)  # (num_nodes_in_batch, embedding_dim)

        length_batches = x.ptr[1:] - x.ptr[:-1]
        time_embeddings = self.time_mlp(t)
        time_embeddings = torch.repeat_interleave(
            time_embeddings, length_batches, dim=0
        )  # (num_nodes_in_batch, embedding_dim)

        embeddings = node_embeddings + time_embeddings  # (num_nodes_in_batch, embedding_dim)

        # TODO: link prediction da rendere efficiente
        # normalized_embeddings = F.normalize(embeddings, dim=-1)
        adj_soft = torch.sigmoid(embeddings @ embeddings.T)

        mask = torch.block_diag(*[torch.ones(i, i) for i in length_batches]).type(torch.bool)
        flattened_adj_soft = adj_soft[mask]
        # normalized_flattened_adj_soft = (flattened_adj_soft + 1) / 2
        # result = torch.stack((flattened_adj_soft, 1 - flattened_adj_soft), dim=-1)

        return flattened_adj_soft


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
