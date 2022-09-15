import torch

from discrete_diffusion.utils import edge_index_to_adj, get_example_from_batch


class Autoencoder(torch.nn.Module):

    def __init__(self, feature_dim, hidden_channels):
        super().__init__()
        self.feature_dim = feature_dim
        # self.encoder = GCNConv(feature_dim, hidden_channels)
        self.lin1 = torch.nn.Linear(10, 128)
        self.mix1 = torch.nn.Linear(10, 10)
        self.lin2 = torch.nn.Linear(128, 128)
        self.mix2 = torch.nn.Linear(10, 10)
        self.lin3 = torch.nn.Linear(128, 128)
        self.mix3 = torch.nn.Linear(10, 10)

    def forward(self, batch):
        # x = batch.x
        edge_index = batch.edge_index
        adj_matrix = edge_index_to_adj(edge_index, num_nodes=10).to(torch.float32)
        z = self.lin1(adj_matrix)
        z = torch.nn.functional.silu(z)
        z = self.mix1(z.T).T
        z = torch.nn.functional.silu(z)
        z = self.lin2(z)
        z = torch.nn.functional.silu(z)
        z = self.mix2(z.T).T
        z = torch.nn.functional.silu(z)
        z = self.lin3(z)
        z = torch.nn.functional.silu(z)
        z = self.mix3(z.T).T
        z = torch.nn.functional.silu(z)
        z = self.out(z)
        return z

        #%%
