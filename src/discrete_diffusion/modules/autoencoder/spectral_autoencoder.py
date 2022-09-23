import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import xitorch
from xitorch.linalg import symeig

from discrete_diffusion.utils import edge_index_to_adj


class SpectralAutoencoder(torch.nn.Module):

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

        self.decode = nn.Linear(10, 50*50)

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

    def decode(self, z):
        out = self.decode(z)
        out = 0.5 * (out + out.T)
        return F.relu(out.reshape(-1, 50, 50))

    def forward(self, batch):
        # dataset embed
        z = self.encode(batch.num_nodes)
        A_pred = self.decode(z)
        A_pred = (A_pred > 0.5).int()

        # compute prediction aeigenvalues
        D_pred = torch.diag(A_pred.sum(dim=0))
        L_pred = D_pred - A_pred
        L_op_pred = xitorch.LinearOperator.m(L_pred)
        evals_pred, _ = symeig(L_op_pred, neig=batch.num_nodes)

        # compute gt aeigenvalues
        A = edge_index_to_adj(batch.edge_index, batch.num_nodes)
        D = torch.diag(A.sum(dim=0))
        L = D - A
        L_op = xitorch.LinearOperator.m(L)
        evals, _ = symeig(L_op, neig=batch.num_nodes)

        loss = torch.linalg.norm(evals - evals_pred)**2
        return loss

#%%
