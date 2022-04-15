import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def compute_cosine_similarities(x, y):
    return F.normalize(x) @ F.normalize(y)


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int):
    adj = torch.zeros((num_nodes, num_nodes)).type_as(edge_index)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    return adj


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:

    return (adj > 0).nonzero().t().type_as(adj)


def get_graph_sizes_from_batch(batch: Batch) -> torch.Tensor:
    return batch.ptr[1:] - batch.ptr[:-1]
