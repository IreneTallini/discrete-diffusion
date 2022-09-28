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
        # self.conv1 = torch_geometric.nn.GCNConv(1, 64)

        self.linear1 = nn.Linear(1, 64, bias=False)
        self.linear2 = nn.Linear(64, 64, bias=False)
        # self.conv2 = torch_geometric.nn.GCNConv(64, 64)
        self.jk = torch_geometric.nn.JumpingKnowledge("cat", 64, num_layers=2)
        self.linear = nn.Linear(2*64, 10)

    def encode(self, batch):
        x, A = batch.x, batch.A
        A = A + torch.eye(x.shape[0])
        D = torch.diag(1/(torch.sqrt(A.sum(dim=-1))))
        L = D @ A @ D
        X = self.dropout(x.unsqueeze(-1))
        X1 = self.linear1(L @ X)
        # X1 = self.conv1(X, edge_index)
        X1 = F.relu(X1)
        # X2 = self.conv2(X1, edge_index)
        X2 = self.linear2(L @ X1)
        X2 = F.relu(X2)
        Xs = [X1, X2]
        X = self.jk(Xs)
        X = torch_geometric.nn.global_mean_pool(X, batch)
        return self.linear(X)

    def forward(self, batch):
        # dataset embed
        emb_1 = self.encode(batch)

        # crete the negative examples (random graphs)
        # random_graph_nx = [nx.generators.fast_gnp_random_graph(n=batch.num_nodes, p=0.5)
        #                    for _ in range(emb_dataset.shape[0])]
        # random_graphs_nx = [torch_geometric.utils.from_networkx(random_graph_nx) ]
        # random_batch = torch_geometric.data.Batch.from_data_list([random_graph])
        # dist = torch.norm(emb[:emb.shape[0] // 2, :] -
        #                   emb[emb.shape[0] // 2:, :]) ** 2
        # loss = torch.max(torch.tensor(0.).type_as(batch.x),
        #                  1. - dist)
        return torch.norm(emb_1)
