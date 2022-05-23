from math import ceil, sqrt
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import scipy.spatial as sp
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx


def compute_self_similarities(x: torch.Tensor) -> torch.Tensor:
    """

    :param x: tensor (num_nodes_in_batch, embedding_dim)

    :return: tensor (num_nodes_in_batch, num_nodes_in_batch)
    """

    x_normalized = F.normalize(x, dim=-1)
    cosine_similarities = x_normalized @ x_normalized.T

    return cosine_similarities


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int):
    adj = torch.zeros((num_nodes, num_nodes)).type_as(edge_index)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    return adj


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:

    return (adj > 0).nonzero().t().type_as(adj)


def get_graph_sizes_from_batch(batch: Batch) -> torch.Tensor:
    return batch.ptr[1:] - batch.ptr[:-1]


def get_data_from_edge_index(edge_index: torch.Tensor, node_features: torch.Tensor) -> Data:
    return Data(x=node_features, edge_index=edge_index, num_nodes=len(node_features))


def get_example_from_batch(batch: Batch, idx: int) -> Data:
    ids = list(range(batch.ptr[idx], batch.ptr[idx + 1]))
    edge_index = []
    min_id = min(ids)
    for edge in batch.edge_index.T:
        if edge[0] in ids:
            edge_index.append(edge - min_id)
    edge_index = torch.stack(edge_index, dim=1)
    node_features = batch.x[batch.ptr[idx] : batch.ptr[idx + 1], :]
    graph = Data(x=node_features, edge_index=edge_index, num_nodes=len(node_features))
    return graph


def unflatten_adj(flattened_adj, num_nodes) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes)).type_as(flattened_adj)
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj[triu_indices[0], triu_indices[1]] = flattened_adj
    adj = adj + adj.T
    return adj


def generate_sampled_graphs_figures(sampled_graphs: List[Data]) -> (plt.Figure, plt.Figure):
    num_samples = len(sampled_graphs)
    side = ceil(sqrt(num_samples))

    fig, axs = plt.subplots(side, side, constrained_layout=True)
    fig_adj, axs_adj = plt.subplots(side, side, constrained_layout=True)
    if side > 1:
        axs = axs.flatten()
        axs_adj = axs_adj.flatten()
    else:
        axs = [axs]
        axs_adj = [axs_adj]

    for i in range(0, num_samples):
        data = sampled_graphs[i]
        axs_adj[i].imshow(edge_index_to_adj(data.edge_index, data.num_nodes).cpu().detach())
        if type(data) == Data:
            data = to_networkx(data)
        nx.draw(data, with_labels=True, ax=axs[i], node_size=0.1)

    return fig, fig_adj
