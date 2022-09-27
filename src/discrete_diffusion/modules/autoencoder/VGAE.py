import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from discrete_diffusion.utils import edge_index_to_adj


class VGAE(nn.Module):
    def __init__(self, feature_dim, hidden_channels):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(feature_dim, hidden_channels)
        self.gcn_mean = GraphConvSparse(hidden_channels, hidden_channels, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden_channels, hidden_channels, activation=lambda x: x)
        self.hidden_channels = hidden_channels

    def encode(self, batch):
        adj = edge_index_to_adj(batch.edge_index, batch.num_nodes)
        x = batch.x
        hidden = self.base_gcn(x, adj)
        mean = self.gcn_mean(hidden, adj)
        logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(batch.num_nodes, self.hidden_channels)
        sampled_z = gaussian_noise * torch.exp(logstd) + mean
        return sampled_z, mean, logstd

    def forward(self, batch):
        # Z is the SAMPLED latent
        Z, mean, logstd = self.encode(batch)
        A_pred = dot_product_decode(Z)
        A_gt = edge_index_to_adj(batch.edge_index, batch.num_nodes)

        logpx_z = - F.binary_cross_entropy_with_logits(input=A_pred, target=A_gt.type_as(A_pred))
        logpz = log_normal_pdf(Z, torch.tensor(0.).type_as(Z), torch.tensor(0.).type_as(Z))
        logqz_x = log_normal_pdf(Z, mean, logstd)

        loss = - (logpx_z + logpz - logqz_x)

        return loss, Z, A_pred


def log_normal_pdf(sample, mean, logvar, dim=1):
    log2pi = torch.log(2. * torch.tensor(np.pi).type_as(sample))  # .as_type(sample)
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi))


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, x, adj):
        # TODO: fallo meglio
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x = torch.mm(x, self.weight)
        x = torch.mm(adj.float(), x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, patch_size*patch_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class GAE(nn.Module):
    def __init__(self, feature_dim, hidden_channels):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(feature_dim, hidden_channels)
        self.gcn_mean = GraphConvSparse(hidden_channels, hidden_channels, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, batch):
        X = batch.x
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred
