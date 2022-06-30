from copy import copy

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions import Categorical
from torch_geometric.data import Batch

from discrete_diffusion.modules.diffusion import Diffusion
from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj


class ContinuousDiffusion(Diffusion):
    def __init__(
        self,
        denoise_fn: DictConfig,
        feature_dim: int,
        diffusion_speed: float,
        timesteps: int,
        threshold_sample: float,
    ):
        super().__init__(denoise_fn, feature_dim, diffusion_speed, timesteps, threshold_sample)

        betas = self.linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def forward_diffusion(self, x_start: Batch, random_timesteps: torch.Tensor) -> Batch:
        """

        :param x_start:
        :param random_timesteps:

        :return:
        """

        x_noisy = (
            self.sqrt_alphas_cumprod[random_timesteps] * x_start + self.sqrt_one_minus_alphas_cumprod[random_timesteps]
        )

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> torch.Tensor:
        """
        Compute q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}

        :param x_start_batch:
        :param t_batch:
        :param x_t_batch:

        :return: tensor (num_possible_edges_batch, 2)
        """
        batch_size = x_start_batch.num_graphs
        t_batch = t_batch.long()

        # (2, 2)
        Q_likelihood = self.Qt[1]
        # (B, 2, 2)
        Q_prior_batch = self.Qt[t_batch - 1]
        Q_evidence_batch = self.Qt[t_batch]

        assert Q_likelihood.shape == (2, 2)
        assert Q_prior_batch.shape == Q_evidence_batch.shape == (batch_size, 2, 2)

        adj_x_t = edge_index_to_adj(x_t_batch.edge_index, x_t_batch.num_nodes)
        adj_x_start = edge_index_to_adj(x_start_batch.edge_index, x_start_batch.num_nodes)

        likelihood = Q_likelihood[:, adj_x_t].permute(1, 2, 0)

        prior = Q_prior_batch[x_start_batch.batch, adj_x_start, :]

        evidence = Q_evidence_batch[x_start_batch.batch, adj_x_start, adj_x_t, None]

        q_backward = (likelihood * prior) / evidence

        length_batches = x_start_batch.ptr[1:] - x_start_batch.ptr[:-1]
        mask = torch.block_diag(*[torch.triu(torch.ones(l, l), diagonal=1) for l in length_batches]).type(torch.bool)

        q_backward_all = q_backward[mask]

        return q_backward_all

    @staticmethod
    def compute_loss(
        q_target: torch.Tensor,
        q_approx: torch.Tensor,
    ):
        cross_ent = F.binary_cross_entropy(q_approx, q_target, reduction="mean")
        ent = F.binary_cross_entropy(q_target, q_target, reduction="mean")
        loss = cross_ent - ent
        return loss


# class GaussianDiffusion(nn.Module):
#     def __init__(
#         self,
#         denoise_fn,
#         *,
#         image_size,
#         channels = 3,
#         timesteps = 1000,
#         loss_type = 'l1',
#         objective = 'pred_noise',
#         beta_schedule = 'cosine'
#     ):
#         super().__init__()
#         assert not (type(self) == GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)
#
#         self.channels = channels
#         self.image_size = image_size
#         self.denoise_fn = denoise_fn
#         self.objective = objective
#
#         if beta_schedule == 'linear':
#             betas = linear_beta_schedule(timesteps)
#         elif beta_schedule == 'cosine':
#             betas = cosine_beta_schedule(timesteps)
#         else:
#             raise ValueError(f'unknown beta schedule {beta_schedule}')
#
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
#
#         timesteps, = betas.shape
#         self.num_timesteps = int(timesteps)
#         self.loss_type = loss_type
#
#         # helper function to register buffer from float64 to float32
#
#         register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
#
#         register_buffer('betas', betas)
#         register_buffer('alphas_cumprod', alphas_cumprod)
#         register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
#
#         # calculations for diffusion q(x_t | x_{t-1}) and others
#
#         register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
#         register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
#         register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
#         register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
#
#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#
#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#
#         # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
#
#         register_buffer('posterior_variance', posterior_variance)
#
#         # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
#
#         register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
#         register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
#
#     def predict_start_from_noise(self, x_t, t, noise):
#         return (
#             extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
#         )
#
#     def q_posterior(self, x_start, x_t, t):
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped
#
#     def p_mean_variance(self, x, t, clip_denoised: bool):
#         model_output = self.denoise_fn(x, t)
#
#         if self.objective == 'pred_noise':
#             x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
#         elif self.objective == 'pred_x0':
#             x_start = model_output
#         else:
#             raise ValueError(f'unknown objective {self.objective}')
#
#         if clip_denoised:
#             x_start.clamp_(-1., 1.)
#
#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
#         return model_mean, posterior_variance, posterior_log_variance
#
#     @torch.no_grad()
#     def p_sample(self, x, t, clip_denoised=True):
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
#         noise = torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
#
#     @torch.no_grad()
#     def p_sample_loop(self, shape):
#         device = self.betas.device
#
#         b = shape[0]
#         img = torch.randn(shape, device=device)
#
#         for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
#             img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
#
#         img = unnormalize_to_zero_to_one(img)
#         return img
#
#     @torch.no_grad()
#     def sample(self, batch_size = 16):
#         image_size = self.image_size
#         channels = self.channels
#         return self.p_sample_loop((batch_size, channels, image_size, image_size))
#
#     @torch.no_grad()
#     def interpolate(self, x1, x2, t = None, lam = 0.5):
#         b, *_, device = *x1.shape, x1.device
#         t = default(t, self.num_timesteps - 1)
#
#         assert x1.shape == x2.shape
#
#         t_batched = torch.stack([torch.tensor(t, device=device)] * b)
#         xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
#
#         img = (1 - lam) * xt1 + lam * xt2
#         for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
#             img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
#
#         return img
#
#     def q_sample(self, x_start, t, noise=None):
#         noise = default(noise, lambda: torch.randn_like(x_start))
#
#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )
#
#     @property
#     def loss_fn(self):
#         if self.loss_type == 'l1':
#             return F.l1_loss
#         elif self.loss_type == 'l2':
#             return F.mse_loss
#         else:
#             raise ValueError(f'invalid loss type {self.loss_type}')
#
#     def p_losses(self, x_start, t, noise = None):
#         b, c, h, w = x_start.shape
#         noise = default(noise, lambda: torch.randn_like(x_start))
#
#         x = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_out = self.denoise_fn(x, t)
#
#         if self.objective == 'pred_noise':
#             target = noise
#         elif self.objective == 'pred_x0':
#             target = x_start
#         else:
#             raise ValueError(f'unknown objective {self.objective}')
#
#         loss = self.loss_fn(model_out, target)
#         return loss
#
#     def forward(self, img, *args, **kwargs):
#         b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
#         assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
#         t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
#
#         img = normalize_to_neg_one_to_one(img)
#         return self.p_losses(img, t, *args, **kwargs)
