from inspect import isfunction

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, *, image_size, channels=1, timesteps=1000, loss_type="l1", betas=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        # alphas = 1.0 - betas
        # alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # to_torch = partial(torch.tensor, dtype=torch.float32)

        # Computing Q_t for each t
        alpha = 1 * 2 / (28 * 27)
        # Qt = torch.empty(self.num_timesteps, 2, 2)
        # for t in range(1, self.num_timesteps + 1):
        #     flip_prob = 0.5 * (1 - (1 - 2 * alpha) ** t)
        #     not_flip_prob = 1 - flip_prob
        #     Q = torch.tensor(
        #         [
        #             [not_flip_prob, flip_prob],
        #             [flip_prob, not_flip_prob],
        #         ],
        #     )
        #     Qt[t - 1] = Q

        Qt = torch.empty(self.num_timesteps, 3, 3)
        for t in range(1, self.num_timesteps + 1):
            flip_prob = 1 - (1 - alpha) ** t
            not_flip_prob = 1 - flip_prob
            Q = torch.tensor(
                [
                    [not_flip_prob, 0, flip_prob],
                    [0, not_flip_prob, flip_prob],
                    [0, 0, 1],
                ],
            )
            Qt[t - 1] = Q
        self.register_buffer("Qt", Qt)

        # self.register_buffer("betas", to_torch(betas))
        # self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        # self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        #
        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        # self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        # self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        # self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        #
        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # self.register_buffer(
        #     "posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        # )
        # self.register_buffer(
        #     "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        # )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_discrete(self, x, t, clip_denoised=True, repeat_noise=False):
        # logits = self.get_q_discrete(self.x0, t, self.x_noisy_tmp)
        logits = self.denoise_fn(x, t)
        sample = torch.distributions.Categorical(logits=logits).sample()
        sample = sample.type(torch.float)
        return sample  # [b,c,h,w]

    @torch.no_grad()
    def p_sample_loop(self, shape):
        # device = self.betas.device

        b = shape[0]
        # img = torch.randn(shape, device=device)
        img = torch.randint(0, 2, shape, dtype=torch.float).type_as(self.Qt[0])

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            times = torch.full((b,), i).type_as(self.Qt[0])
            img = self.p_sample_discrete(img, times)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_q_discrete(self, x_start, t, x_t, noise=None):
        # Expression for q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}
        Q_likelihood = self.Qt[0]  # [b, num_cat, num_cat]
        Q_prior = self.Qt[t - 1]
        Q_evidence = self.Qt[t]
        x_t_one_hot = torch.nn.functional.one_hot(x_t.type(torch.int64), num_classes=3).type(torch.float)
        x_start_one_hot = torch.nn.functional.one_hot(x_start.type(torch.int64), num_classes=3).type(torch.float)
        likelihood = torch.einsum("bchwk, pk -> bchwp", x_t_one_hot, Q_likelihood)
        prior = torch.einsum("bchwk, bkp -> bchwp", x_start_one_hot, Q_prior)
        evidence = torch.einsum("bchwk, bkl, bchwl -> bchw", x_start_one_hot, Q_evidence, x_t_one_hot)
        q_backward = (likelihood * prior) / evidence.unsqueeze(-1)  # [b,c,h,w,num_cat]
        return q_backward

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        # noise = default(noise, lambda: torch.randn_like(x_start))

        # For each categorical value in x_start, the corresponding
        # row in Q is the vector of q(xt | x0)
        Q_batch = self.Qt[t]  # [b, num_cat, num_cat]
        x_start_one_hot = torch.nn.functional.one_hot(x_start.type(torch.int64), num_classes=3).type(torch.float)
        q = torch.einsum("bchwk, bkp -> bchwp", x_start_one_hot, Q_batch)

        x_noisy = torch.distributions.Categorical(q).sample().type(torch.float)  # [b,c,h,w]

        q_noisy = self.get_q_discrete(x_start=x_start, t=t, x_t=x_noisy)  # [b,c,h,w,num_cat]
        q_recon = self.denoise_fn(x_noisy, t)
        # wandb.log({"q noisy": q_noisy.abs().sum()})
        # wandb.log({"q recon": q_recon.abs().sum()})
        # q_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # q_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == "l1":
            loss = (noise - q_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, q_recon)
        elif self.loss_type == "kl_div":
            loss = F.cross_entropy(
                q_recon.permute(0, 4, 1, 2, 3), q_noisy.permute(0, 4, 1, 2, 3)
            )  # kl_div(q_noisy, q_recon, reduction="none")
            # loss = loss.sum((-1, -2, -3), dtype=float)
            # loss = loss.mean()
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, _, h, w, device, img_size, = (
            *x.shape,
            x.device,
            self.image_size,
        )
        assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
