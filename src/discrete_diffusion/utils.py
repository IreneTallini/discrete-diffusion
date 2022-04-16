import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def compute_cosine_similarities(x, y):
    cosine_sim = F.normalize(x) @ F.normalize(y, dim=0)
    # Ensures cosine sim is between 1 and -1
    cosine_sim = cosine_sim.clamp(min=-0.99, max=0.99)
    return cosine_sim


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int):
    adj = torch.zeros((num_nodes, num_nodes)).type_as(edge_index)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    return adj


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:

    return (adj > 0).nonzero().t().type_as(adj)


def get_graph_sizes_from_batch(batch: Batch) -> torch.Tensor:
    return batch.ptr[1:] - batch.ptr[:-1]
