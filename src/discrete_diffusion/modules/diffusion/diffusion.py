import hydra
import torch
from torch import nn

from discrete_diffusion.modules.diffusion import sde_lib


class Diffusion(nn.Module):

    def __init__(self, sde, score_net, sampler, feature_len, n_samples=1,
                 continuous=True, likelihood_weighting=True,
                 reduce_mean=False):
        super(Diffusion, self).__init__()
        self.sde = hydra.utils.instantiate(sde, _recursive_=False)
        self.score_net = hydra.utils.instantiate(score_net, _recursive_=False)
        self.sampler = hydra.utils.instantiate(sampler, sde=self.sde, score_fn=self.score_fn,
                                               shape=(n_samples, self.score_net.feature_dim, feature_len, 1),
                                               _recursive_=False)
        self.continuous = continuous
        self.likelihood_weighting = likelihood_weighting
        self.reduce_mean = reduce_mean

    def forward(self, batch):
        return self.sde_loss_fn(batch)

    def sample(self, device='cpu'):
        return self.sampler.sample(device=device)

    def score_fn(self, x, t):

        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            # Scale neural network output by standard deviation and flip sign
            if self.continuous or isinstance(self.sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = self.score_net(x, labels)
                std = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (self.sde.N - 1)
                score = self.score_net(x, labels)
                std = self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

        elif isinstance(self.sde, sde_lib.VESDE):
            if self.continuous:
                labels = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = self.sde.T - t
                labels *= self.sde.N - 1
                labels = torch.round(labels).long()

            score = self.score_net(x, labels)
            return score

        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

    def sde_loss_fn(self, batch, eps=1e-5):
        """Compute the loss function.
        Args:
          batch: A mini-batch of training data.
          eps: A `float` number. The smallest time step to sample from.
        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        # TODO: change dimensions inside the function to work with latent node embeddings

        reduce_op = torch.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = self.score_fn(perturbed_data, t)

        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss
