import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from discrete_diffusion.utils import edge_index_to_adj, get_example_from_batch


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.Q = nn.Linear(channels, channels)
        self.K = nn.Linear(channels, channels)
        self.V = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        h = F.softmax(q @ k.T, dim=-1) @ v
        h = self.out(h)
        # w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        # w = torch.reshape(w, (B, H, W, H * W))
        # w = F.softmax(w, dim=-1)
        # w = torch.reshape(w, (B, H, W, H, W))
        # h = torch.einsum('bhwij,bcij->bchw', w, v)
        # h = self.NIN_3(h)
        return x + h


def log_normal_pdf(sample, mean, logvar, dim=1):
    log2pi = torch.log(2. * torch.tensor(np.pi).type_as(sample))  # .as_type(sample)
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi))


class Autoencoder(torch.nn.Module):

    def __init__(self, feature_dim, hidden_channels):
        super().__init__()
        self.feature_dim = feature_dim

        # encoder
        self.encoder = torch.nn.Sequential(
            nn.Linear(50, 256),
            nn.SiLU(),
            AttnBlock(256),
            nn.SiLU(),
            AttnBlock(256),
            nn.SiLU(),
            AttnBlock(256),
            nn.SiLU())
        self.enc_out = torch.nn.Linear(50, 20)
            # nn.Linear(128, 256))
            # n.SiLU(),

        self.dec_in = torch.nn.Linear(20, 50)
        self.decoder = torch.nn.Sequential(
            nn.Linear(256, 50))

    def encode(self, batch):
        edge_index = batch.edge_index
        adj_matrix = edge_index_to_adj(edge_index, num_nodes=50).to(torch.float32)
        emb = self.encoder(adj_matrix)
        emb = self.enc_out(emb.T).T
        # mean, logvar = torch.split(self.encoder(adj_matrix), split_size_or_sections=emb.shape[-1] // 2, dim=1)
        return emb  # mean, logvar

    def decode(self, z):
        z = self.dec_in(z.T).T
        z = F.silu(z)
        return self.decoder(z)

    # def reparameterize(self, mean, logvar):
    #    eps = torch.randn(size=mean.shape).type_as(mean)
    #    return eps * torch.exp(logvar * .5) + mean

    def forward(self, batch):
        # x = torch.rand((10, 10)).t
        # ype_as(batch.edge_index).to(torch.float32)
        # x.requires_grad = True
        # mean, logvar = self.encode(batch)

        z = self.encode(batch)
        # z = self.reparameterize(mean, logvar)

        x = self.decode(z)
        # # cross_ent = torch.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        # # logpx_z = -torch.reduce_sum(cross_ent, axis=[1, 2, 3])

        gt_adjs = edge_index_to_adj(batch.edge_index, len(batch.batch))
        # logpx_z = -F.binary_cross_entropy_with_logits(input=x, target=gt_adjs.type_as(x))
        # logpx_z = -torch.reduce_sum(cross_ent, axis=[1, 2, 3])

        logpx_z = -0.5 * torch.norm(gt_adjs - x) ** 2
        # logpz = log_normal_pdf(z, torch.tensor(0.).type_as(z), torch.tensor(0.).type_as(z))
        # logqz_x = log_normal_pdf(z, mean, logvar)

        return -torch.mean(logpx_z), z, x #+ logpz - logqz_x), z, x

#%%
