from math import pi

import hydra.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from omegaconf import DictConfig
from tqdm import tqdm

from discrete_diffusion.modules.transformer.ops import filter_logits
from discrete_diffusion.modules.transformer.transformer import Transformer
from discrete_diffusion.utils import edge_index_to_adj, flatten_batch, unflatten_adj


def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


def get_normal(*shape, std=0.01):
    w = torch.empty(shape)
    nn.init.normal_(w, std=std)
    return w


class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        return self.pos_emb


class AutoencoderStochasticDecoder(torch.nn.Module):
    def __init__(self, node_embedder: DictConfig, feature_dim=64, enc_channels=64,
                 latent_channels=64, dec_channels=64, max_num_nodes=10):
        super().__init__()
        self.enc_channels = enc_channels
        self.latent_channels = latent_channels
        self.dec_channels = dec_channels
        self.feature_dim = feature_dim
        self.max_num_nodes = max_num_nodes[0]

        # encoder
        self.node_embedder = hydra.utils.instantiate(node_embedder, feature_dim=feature_dim, _recursive_=False)
        # self.enc_dropout = torch.nn.Dropout(p=0.5)
        # self.enc_conv1 = torch_geometric.nn.GCNConv(self.feature_dim, enc_channels)
        # self.enc_conv2 = torch_geometric.nn.GCNConv(enc_channels, enc_chan    nels)
        # mlp1 = torch.nn.Sequential(torch.nn.Linear(self.feature_dim, self.enc_channels),
        #                           torch.nn.ReLU(),
        #                           torch.nn.Linear(self.enc_channels, self.enc_channels))
        # mlp2 = torch.nn.Sequential(torch.nn.Linear(self.enc_channels, self.enc_channels),
        #                            torch.nn.ReLU(),
        #                            torch.nn.Linear(self.enc_channels, self.enc_channels))
        # self.enc_conv1 = hydra.utils.instantiate(gnn, in_channels=self.feature_dim)
        # self.enc_conv2 = hydra.utils.instantiate(gnn, in_channels=enc_channels)
        # self.enc_jk = torch_geometric.nn.JumpingKnowledge("cat", enc_channels, num_layers=2)
        self.enc_linear = nn.Linear(enc_channels, latent_channels)

        # decoder
        self.conditioneer = nn.Linear(latent_channels, dec_channels)
        self.dec_embedder = nn.Embedding(num_embeddings=2, embedding_dim=dec_channels)
        self.num_edges = int(self.max_num_nodes * (self.max_num_nodes - 1) / 2)
        self.pos_emb = PositionEmbedding(input_shape=(self.num_edges,), width=dec_channels, init_scale=1.0)
        if self.max_num_nodes % 2 == 0:
            num_blocks = self.max_num_nodes - 1
        else:
            num_blocks = self.max_num_nodes
        self.transformer = Transformer(n_in=dec_channels, n_ctx=self.num_edges, n_head=1, n_depth=2,
                                       attn_dropout=0.0, resid_dropout=0.0,
                                       afn='quick_gelu', scale=True, mask=True,
                                       zero_out=False, init_scale=0.7, res_scale=False,
                                       m_attn=0.25, m_mlp=1.0,
                                       checkpoint_attn=0, checkpoint_mlp=0, checkpoint_res=1,
                                       attn_order=0, blocks=num_blocks, spread=None,
                                       encoder_dims=0, prime_len=0)
        self.dec_out = nn.Linear(dec_channels, 2)

    def encode(self, batch):
        # x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        # if len(x.shape) == 1:
        #     x = x.unsqueeze(-1)
        # X = self.enc_dropout(x)
        # X1 = self.enc_conv1(X, edge_index)
        # X2 = self.enc_conv2(X1, edge_index)
        # Xs = [X1, X2]
        # X = self.enc_jk(Xs)
        X = self.node_embedder(batch)
        X = torch_geometric.nn.global_mean_pool(X, batch.batch)
        return self.enc_linear(X)

    def get_emb(self, sample_t, n_samples, x, z):
        if sample_t == 0:
            # Fill in start token
            x = torch.empty(n_samples, 1, self.dec_channels, device=z.device)
            z = self.conditioneer(z)
            x[:, 0] = z
        else:
            x = self.dec_embedder(x)
        assert x.shape == (n_samples, 1, self.dec_channels)
        x = x + self.pos_emb()[sample_t:sample_t + 1]  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.dec_channels)
        return x

    def decode_sample(self, z, n_samples, temp=1.0, top_k=0, top_p=0, fp16=False):
        with torch.no_grad():
            xs, x = [], None
            for sample_t in tqdm(range(self.num_edges)):
                x = self.get_emb(sample_t, n_samples, x, z)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=None, sample=True, fp16=fp16) # Transformer
                assert x.shape == (n_samples, 1, self.dec_channels)
                x = self.dec_out(x)  # Predictions
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = torch.distributions.Categorical(logits=x).sample() # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())
            del x
            self.transformer.del_cache()

            x = torch.cat(xs, dim=1)
            # x = self.postprocess(x, sample_tokens)
            return x

    def decode_train(self, z, batch):
        _, edge_index, _ = batch.x, batch.edge_index, batch.batch
        A_batch = edge_index_to_adj(edge_index, num_nodes=batch.num_nodes)
        batch_size = len(batch.ptr) - 1
        A_flat_batch = flatten_batch(batch, A_batch).reshape(batch_size, -1) # Reshape rispetto alla batch size
        x_t = A_flat_batch.reshape(batch_size, -1)
        x = self.dec_embedder(A_flat_batch)  # [B, N, C]
        x = roll(x, 1)  # Shift by 1, and fill in start token
        # Match dim of z with dim of 0-1 embeddings
        z = self.conditioneer(z)
        # Add z embedding as first token
        x[:, 0, :] = z
        # Add positional embedding
        x = x + self.pos_emb()  # Pos emb and dropout
        # Transformer implementing the autoregressive part
        x = self.transformer(x)  # Transformer
        # Reduce from [B, N, C] to [B, N, 2]
        x = self.dec_out(x)  # Predictions
        # La loss non deve essere autoregressiva?
        loss = F.cross_entropy(x.view(-1, 2), x_t.view(-1)) / np.log(2.)  # Loss
        return loss

        # data = torch.cat((data, enc_pos), dim = -1)
        #  # Target

    def forward(self, batch):
        # dataset embed
        z = self.encode(batch)
        loss = self.decode_train(z, batch)
        return loss, z
#%%
