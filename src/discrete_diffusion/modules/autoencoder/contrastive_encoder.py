import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from discrete_diffusion.utils import edge_index_to_adj


class ContrastiveEncoder(torch.nn.Module):

    def __init__(self, feature_dim, hidden_channels):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = torch_geometric.nn.GCNConv(1, 64)
        # self.W1 = nn.Linear(64, 64, bias=False)
        # self.W2 = nn.Linear(64, 64, bias=False)
        self.conv2 = torch_geometric.nn.GCNConv(64, 64)
        self.jk = torch_geometric.nn.JumpingKnowledge("cat", 64, num_layers=2)
        self.linear = nn.Linear(2*64, 10)

    def encode(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        x = self.dropout(x.unsqueeze(-1))
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        xs = [x1, x2]
        x = self.jk(xs)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return self.linear(x)

    def forward(self, batch):
        # dataset embed
        emb_dataset = self.encode(batch)

        # crete the negative examples (random graphs)
        random_graph_nx = [nx.generators.fast_gnp_random_graph(n=batch.num_nodes, p=0.5)
                           for _ in range(emb_dataset.shape[0])]
        random_graphs_nx = [torch_geometric.utils.from_networkx(random_graph_nx) ]

        random_batch = torch_geometric.data.Batch.from_data_list([random_graph])

        dist = torch.norm(emb_positive[:emb_positive.shape[0] // 2, :] -
                          emb_positive[emb_positive.shape[0] // 2:, :]) ** 2
        loss = torch.max(torch.tensor(0.).type_as(batch.x),
                         1. - dist)
        return loss
#%%
