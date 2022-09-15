import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from discrete_diffusion.utils import edge_index_to_adj, get_example_from_batch


class Autoencoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_channels):
        super().__init__()
        self.feature_dim = feature_dim
        # self.encoder = GCNConv(feature_dim, hidden_channels)
        self.out = torch.nn.Linear(10, 128)

    def forward(self, batch):
        # x = batch.x
        edge_index = batch.edge_index
        z = self.out(edge_index_to_adj(edge_index, num_nodes=10).to(torch.float32))  # x.unsqueeze(dim=1), edge_index)
        return z
