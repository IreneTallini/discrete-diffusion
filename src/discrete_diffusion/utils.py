import torch


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int):

    adj = torch.zeros((num_nodes, num_nodes)).long()
    adj[edge_index] = 1
    return adj


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:

    return (adj > 0).nonzero().t()
