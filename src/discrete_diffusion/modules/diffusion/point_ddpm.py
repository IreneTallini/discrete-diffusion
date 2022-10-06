# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.
This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import functools

import hydra
import torch
import torch.nn as nn

from discrete_diffusion.modules.diffusion import layers

ResnetBlockPointDDPM = layers.ResnetBlockPointDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
default_initializer = layers.default_init
conv1x1 = layers.ddpm_conv1x1


class PointDDPM(nn.Module):
    def __init__(self, feature_dim, nonlinearity, hidden_dim, ch_mult, num_res_blocks,
                 attn_resolutions, dropout=0.1,  conditional=True):
        super().__init__()
        self.act = act = hydra.utils.instantiate(nonlinearity, _recursive_=False)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        # self.all_resolutions = all_resolutions = [config.data.image_size `//` (2 ** i) for i in range(num_resolutions)]
        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional
        ResnetBlock = functools.partial(ResnetBlockPointDDPM, act=act, temb_dim=4 * hidden_dim, dropout=dropout)
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(hidden_dim, hidden_dim * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(hidden_dim * 4, hidden_dim * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        # Downsampling block
        modules.append(conv1x1(feature_dim, hidden_dim))
        hs_c = [hidden_dim]
        # hs_c = []
        in_ch = hidden_dim
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = hidden_dim * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                # if all_resolutions[i_level] in attn_resolutions:
                #     modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks):
                out_ch = hidden_dim * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            # if all_resolutions[i_level] in attn_resolutions:
            #    modules.append(AttnBlock(channels=in_ch))

        # assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv1x1(in_ch, feature_dim, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.hidden_dim)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        h = x

        # Downsampling block
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # if h.shape[-1] in self.attn_resolutions:
                #    h = modules[m_idx](h)
                #    m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            # if h.shape[-1] in self.attn_resolutions:
            #    h = modules[m_idx](h)
            #    m_idx += 1

        #  assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        return h
