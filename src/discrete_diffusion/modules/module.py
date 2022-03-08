"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import functools

import torch
import torch.nn as nn

from . import layers, utils
from .layers import ddpm_conv1x1 as conv1x1

ResnetBlockGraphDDPM = layers.ResnetBlockGraphDDPM
get_act = layers.get_act
default_initializer = layers.default_init


class GraphDDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        # self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        self.num_resolutions = num_resolutions = len(ch_mult)

        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockGraphDDPM, act=act, temb_dim=4 * nf, dropout=dropout)

        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.model.centered
        channels = config.model.num_channels

        # Downsampling block
        modules.append(conv1x1(channels, nf))
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                modules.append(AttnBlock(channels=in_ch))

        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv1x1(in_ch, channels, init_scale=0.0))
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.0

        # Downsampling block
        h = modules[m_idx](h)
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](h, temb)
                m_idx += 1
                h = modules[m_idx](h)
                m_idx += 1

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)

        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas

        return h


# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
# Example module for the template
class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential()
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.model(x)
        # [batch_size, 32 * 7 * 7]
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
