import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.nn import CosineSimilarity
import scipy.spatial as sp


def compute_self_similarities(x: torch.Tensor) -> torch.Tensor:
    """

    :param x: tensor (num_nodes_in_batch, embedding_dim)

    :return: tensor (num_nodes_in_batch, num_nodes_in_batch)
    """

    x_normalized = F.normalize(x, dim=-1)
    cosine_similarities = x_normalized @ x_normalized.T

    # Ensures cosine sim is between 1 and -1
    tol = 1e-6
    # max = 1 - tol
    # min = -(1 - tol)
    cosine_similarities = cosine_similarities.clamp(min=-(1 - tol), max=1 - tol)

    similarities_normalized = (cosine_similarities + 1) / (2)

    return similarities_normalized


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int):
    adj = torch.zeros((num_nodes, num_nodes)).type_as(edge_index)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    return adj


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:

    return (adj > 0).nonzero().t().type_as(adj)


def get_graph_sizes_from_batch(batch: Batch) -> torch.Tensor:
    return batch.ptr[1:] - batch.ptr[:-1]
