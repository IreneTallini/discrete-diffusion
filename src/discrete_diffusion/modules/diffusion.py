from inspect import isfunction

# import matplotlib.pyplot as plt
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj

from discrete_diffusion.utils import edge_index_to_adj, adj_to_edge_index


class Diffusion(nn.Module):
    def __init__(self, denoise_fn: DictConfig, feature_dim, diffusion_speed, timesteps=1000):
        super().__init__()
        self.denoise_fn = instantiate(denoise_fn, cfg=denoise_fn, feature_dim=feature_dim, _recursive_=False)
        self.num_timesteps = int(timesteps)
        self.diffusion_speed = diffusion_speed
        self.Qt = self.get_Qt()

    def get_Qt(self):
        Qt = torch.empty(self.num_timesteps, 2, 2)
        for t in range(1, self.num_timesteps + 1):
            flip_prob = 0.5 * (1 - (1 - 2 * self.diffusion_speed) ** t)
            not_flip_prob = 1 - flip_prob
            Q = torch.tensor(
                [
                    [not_flip_prob, flip_prob],
                    [flip_prob, not_flip_prob],
                ],
            )
            Qt[t - 1] = Q
        return Qt

    @torch.no_grad()
    def p_sample_discrete(self, x, t):
        logits = self.denoise_fn(x, t)
        sample = torch.distributions.Categorical(logits=logits).sample()
        sample = sample.type(torch.float)
        return sample  # [b,c,h,w]

    @torch.no_grad()
    def p_sample_loop(self, shape):

        b = shape[0]
        img = torch.full(shape, 2, dtype=torch.float).type_as(self.Qt[0])

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            times = torch.full((b,), i).type_as(self.Qt[0])
            img = self.p_sample_discrete(img, times)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> List[torch.Tensor]:
        Qt = self.Qt

        # Expression for q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}
        Q_likelihood = Qt[0]  # [b, num_cat, num_cat]
        Q_prior_batch = Qt[t_batch - 1]
        Q_evidence_batch = Qt[t_batch]

        x_start_data_list: List[Data] = x_start_batch.to_data_list()
        x_t_data_list: List[Data] = x_t_batch.to_data_list()
        batch_size = t_batch.shape[0]

        q_backward_list = []
        for b in range(batch_size):

            x_start, x_t, Q_prior, Q_evidence = (
                x_start_data_list[b],
                x_t_data_list[b],
                Q_prior_batch[b],
                Q_evidence_batch[b],
            )
            # x_t_one_hot = torch.nn.functional.one_hot(x_t.type(torch.int64), num_classes=3).type(torch.float)
            # x_start_one_hot = torch.nn.functional.one_hot(x_start.type(torch.int64), num_classes=3).type(torch.float)

            adj_x_t = edge_index_to_adj(x_t.edge_index, x_t.num_nodes)
            adj_x_start = edge_index_to_adj(x_start.edge_index, x_start.num_nodes)

            # TODO: check correctness
            likelihood = Q_likelihood[adj_x_t]
            prior = Q_prior[adj_x_start]
            evidence = Q_evidence[adj_x_start, adj_x_t]

            # likelihood = torch.einsum("bchwk, pk -> bchwp", x_t_one_hot, Q_likelihood)
            # prior = torch.einsum("bchwk, bkp -> bchwp", x_start_one_hot, Q_prior)
            # evidence = torch.einsum("bchwk, bkl, bchwl -> bchw", x_start_one_hot, Q_evidence, x_t_one_hot)

            q_backward = (likelihood * prior) / evidence.unsqueeze(-1)  # [n, n, 2]
            q_backward_list.append(q_backward)

        return q_backward_list

    def loss(self, q_noisy, q_recon):
        loss = F.cross_entropy(q_recon.permute(0, 4, 1, 2, 3), q_noisy.permute(0, 4, 1, 2, 3))
        return loss

    def forward_diffusion(self, x_start: Batch, t: torch.Tensor):
        """

        :param x_start:
        :param t:
        :return:
        """
        Q_batch = self.Qt[t]  # [b, n, n]

        data_list: List[Data] = x_start.to_data_list()

        for b, data in enumerate(data_list):

            adj = edge_index_to_adj(data.edge_index, data.num_nodes)
            Q = Q_batch[b]

            q = Q[adj]
            x_noisy = torch.distributions.Categorical(q).sample()  # [n, n]

            new_edge_index = adj_to_edge_index(x_noisy)
            data.edge_index = new_edge_index

        x_noisy = Batch.from_data_list(data_list)

        return x_noisy

    def forward(self, x_start: Batch, *args, **kwargs):
        batch_size = x_start.ptr.shape[0] - 1
        t = torch.randint(0, self.num_timesteps, (batch_size,)).type_as(x_start["edge_index"])

        x_noisy = self.forward_diffusion(x_start, t)
        q_noisy = self.backward_diffusion(x_start_batch=x_start, t_batch=t, x_t_batch=x_noisy)  # [b,c,h,w,num_cat]
        q_approx = self.denoise_fn(x_noisy, t)

        loss = self.loss(q_noisy, q_approx)

        return loss
