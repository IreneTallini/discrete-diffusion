from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
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
        self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale)).to('cuda')

    def forward(self):
        return self.pos_emb


class AutoencoderStochasticDecoder(torch.nn.Module):
    def __init__(self, feature_dim=64, enc_channels=64, latent_channels=64, dec_channels=64):
        super().__init__()
        self.enc_channels = enc_channels
        self.latent_channels = latent_channels
        self.dec_channels = dec_channels

        # encoder
        self.enc_dropout = torch.nn.Dropout(p=0.5)
        # self.enc_conv1 = torch_geometric.nn.GCNConv(1, enc_channels)
        # self.enc_conv2 = torch_geometric.nn.GCNConv(enc_channels, enc_channels)
        nn = torch.nn.Sequential(torch.nn.Linear(1, 64),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(64, 64))
        self.enc_conv1 = torch_geometric.nn.GINConv(nn)
        self.enc_conv2 = torch_geometric.nn.GINConv(nn)
        self.enc_jk = torch_geometric.nn.JumpingKnowledge("cat", enc_channels, num_layers=2)
        self.enc_linear = nn.Linear(2*enc_channels, latent_channels)

        # decoder
        self.conditioneer = nn.Linear(latent_channels, dec_channels)
        self.dec_embedder = nn.Embedding(num_embeddings=2, embedding_dim=dec_channels)
        self.pos_emb = PositionEmbedding(input_shape=(1225,), width=128, init_scale=1.0)
        self.transformer = Transformer(n_in=128, n_ctx=1225, n_head=1, n_depth=2,
                                       attn_dropout=0.0, resid_dropout=0.0,
                                       afn='quick_gelu', scale=True, mask=True,
                                       zero_out=False, init_scale=0.7, res_scale=False,
                                       m_attn=0.25, m_mlp=1.0,
                                       checkpoint_attn=0, checkpoint_mlp=0, checkpoint_res=1,
                                       attn_order=0, blocks=49, spread=None,
                                       encoder_dims=0, prime_len=0)
        self.dec_out = nn.Linear(128, 2)

    def encode(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        X = self.enc_dropout(x.unsqueeze(-1))
        X1 = self.enc_conv1(X, edge_index)
        # X1 = F.relu(X1)
        X2 = self.enc_conv2(X1, edge_index)
        #X2 = F.relu(X2)
        # Xs = [X1, X2]
        # X = self.enc_jk(Xs)
        X = torch_geometric.nn.global_mean_pool(X, batch)
        return self.enc_linear(X)

    def get_emb(self, sample_t, n_samples, x, z):
        if sample_t == 0:
            # Fill in start token
            x = torch.empty(n_samples, 1, self.dec_channels).cuda()
            z = self.conditioneer(z)
            x[:, 0] = z
        else:
            x = self.dec_embedder(x)
        assert x.shape == (n_samples, 1, self.dec_channels)
        x = x + self.pos_emb()[sample_t:sample_t + 1]  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.dec_channels)
        return x

    def decode_sample(self, z, n_samples, sample_tokens=1225, temp=1.0, top_k=0, top_p=0, fp16=False):
        with torch.no_grad():
            xs, x = [], None
            for sample_t in tqdm(range(sample_tokens)):
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
        A = edge_index_to_adj(edge_index, num_nodes=batch.num_nodes)
        a = flatten_batch(batch, A).reshape(1, -1)
        x_t = a
        x = self.dec_embedder(a)
        x = roll(x, 1)  # Shift by 1, and fill in start token
        z = self.conditioneer(z)
        x[:, 0] = z
        # x = x.permute(0, 2, 1)
        x = x + self.pos_emb()  # Pos emb and dropout
        x = self.transformer(x)  # Transformer
        x = self.dec_out(x)  # Predictions
        loss = F.cross_entropy(x.view(-1, 2), x_t.view(-1)) / np.log(2.)  # Loss
        return loss

        # data = torch.cat((data, enc_pos), dim = -1)
        #  # Target

    def forward(self, batch):
        # dataset embed
        z = self.encode(batch)
        loss = self.decode_train(z, batch)
        return loss
#%%
