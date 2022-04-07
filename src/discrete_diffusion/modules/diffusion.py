from inspect import isfunction

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(self, cfg, feature_dim, diffusion_speed, timesteps=1000):
        super().__init__()
        self.cfg = cfg
        self.denoise_fn = instantiate(self.cfg.denoise_fn, feature_dim=feature_dim)
        self.num_timesteps = int(timesteps)
        self.diffusion_speed = diffusion_speed

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

    def backward_diffusion(self, x_start: Batch, t, x_t):
        # Expression for q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}
        Qt = self.get_Qt()
        Q_likelihood = Qt[0]  # [b, num_cat, num_cat]
        Q_prior = Qt[t - 1]
        Q_evidence = Qt[t]
        x_t_one_hot = torch.nn.functional.one_hot(x_t.type(torch.int64), num_classes=3).type(torch.float)
        x_start_one_hot = torch.nn.functional.one_hot(x_start.type(torch.int64), num_classes=3).type(torch.float)
        likelihood = torch.einsum("bchwk, pk -> bchwp", x_t_one_hot, Q_likelihood)
        prior = torch.einsum("bchwk, bkp -> bchwp", x_start_one_hot, Q_prior)
        evidence = torch.einsum("bchwk, bkl, bchwl -> bchw", x_start_one_hot, Q_evidence, x_t_one_hot)
        q_backward = (likelihood * prior) / evidence.unsqueeze(-1)  # [b,c,h,w,num_cat]
        return q_backward

    def loss(self, q_noisy, q_recon):
        loss = F.cross_entropy(q_recon.permute(0, 4, 1, 2, 3), q_noisy.permute(0, 4, 1, 2, 3))
        return loss

    def forward_diffusion(self, x_start: Batch, t: torch.Tensor):
        """

        :param x_start:
        :param t:
        :return:
        """
        Qt = self.get_Qt()
        Q_batch = Qt[t]  # [b, num_cat, num_cat]
        num_nodes_list = x_start.ptr[1:] - x_start.ptr[:-1]
        num_blocks = len(num_nodes_list)

        x_start_one_hot = torch.nn.functional.one_hot(x_start.type(torch.int64), num_classes=2).type(torch.float)
        q = torch.einsum("bchwk, bkp -> bchwp", x_start_one_hot, Q_batch)

        x_noisy = torch.distributions.Categorical(q).sample().type(torch.float)  # [b,c,h,w]
        return x_noisy

    def forward(self, x_start: Batch, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        x_noisy = self.forward_diffusion(x_start, t)
        q_noisy = self.backward_diffusion(x_start=x_start, t=t, x_t=x_noisy)  # [b,c,h,w,num_cat]
        q_approx = self.denoise_fn(x_noisy, t)

        loss = self.loss(q_noisy, q_approx)

        return loss
