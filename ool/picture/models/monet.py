"""
Reimplementation of MONet
"MONet: Unsupervised Scene Decomposition and Representation"
Christopher P. Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra,
Irina Higgins, Matt Botvinick and Alexander Lerchner
https://arxiv.org/abs/1901.11390


Somewhat based on implementation from
https://github.com/baudm/MONet-pytorch
"""

import itertools

import ipdb as ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from .ool_base import OOLBase


class ComponentEncoder(nn.Sequential):
    """
    The paper seems to to target the cell_width of 12x12 pixels.
    """

    def __init__(self, in_shape, z_dim=16):
        inc, h, w = in_shape
        h = ((((h + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
        w = ((((w + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
        super(ComponentEncoder, self).__init__(
            nn.Conv2d(inc + 1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )
        self.z_dim = z_dim

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = super(ComponentEncoder, self).forward(x)
        mu, logstd = x[:, :self.z_dim], x[:, self.z_dim:]
        return mu, logstd.exp()


class ComponentDecoder(nn.Sequential):
    def __init__(self, z_dim, out_shape):
        super(ComponentDecoder, self).__init__(
            nn.Conv2d(z_dim + 2, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, out_shape[0] + 1, kernel_size=1, stride=1, padding=0),
        )
        self.out_shape = out_shape

    def forward(self, x):
        h, w = self.out_shape[-2:]
        h += 8
        w += 8
        x = x.view(x.size(0), -1, 1, 1).repeat(1, 1, h, w)
        hs = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        ws = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        c = torch.stack(torch.meshgrid(hs, ws)).view(1, 2, h, w).repeat(
            x.size(0), 1, 1, 1)
        x = torch.cat([x, c], dim=1)
        x = super(ComponentDecoder, self).forward(x)
        img = torch.sigmoid(x[:, :self.out_shape[0]])
        msk = x[:, -1:]
        return img, msk


class AttentionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def comb(self, x, y):
        if x.shape == y.shape:
            return torch.cat([x, y], dim=1)
        return torch.cat([
            F.pad(x, (0, y.size(-1) - x.size(-1), 0, y.size(-2) - x.size(-2)), 'constant', 0),
            y
        ], dim=1)

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else self.comb(*inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2., mode='nearest')
        return (x, skip) if downsampling else x


class Attention(nn.Module):
    def __init__(self, numb, in_shape, ngf=64):
        super(Attention, self).__init__()
        c, h, w = in_shape
        x = torch.zeros(1, c+1, h, w)
        self.downblocks = nn.ModuleList([
            AttentionBlock(c + 1, ngf, resize=True)  # Fist
        ])
        x = self.downblocks[-1](x)[0]
        # print(x.shape)
        upblocks = [AttentionBlock(2 * ngf, ngf, resize=False)]  # Last
        for i in range(1, numb - 1):
            self.downblocks.append(
                AttentionBlock(ngf * 2 ** (i - 1), ngf * min(2 ** i, 8), resize=True)
            )
            x = self.downblocks[-1](x)[0]
            # print(x.shape)
            upblocks.append(
                AttentionBlock(2 * ngf * min(2 ** i, 8), ngf * min(2 ** (i - 1), 8), resize=True)
            )
        self.downblocks.append(
            AttentionBlock(ngf * min(2 ** (numb - 1), 8), ngf * min(2 ** (numb - 1), 8), resize=False)
        )
        x = self.downblocks[-1](x)[0]
        # print(x.shape)
        upblocks.append(
            AttentionBlock(2 * ngf * min(2 ** (numb - 1), 8), ngf * min(2 ** (numb - 1), 8), resize=True)
        )
        self.upblocks = nn.ModuleList(list(reversed(upblocks)))
        inc = np.product(x.shape)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inc, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, inc),
            nn.ReLU(),
        )
        self.output = nn.Conv2d(ngf, 1, kernel_size=1)

    def forward(self, x, log_sk):
        x = torch.cat((x, log_sk), dim=1)
        skips = []
        for l in self.downblocks:
            x, skip = l(x)
            skips.append(skip)
            # print('down', x.shape, skip.shape)

        x = self.mlp(x).view(skips[-1].shape)

        for l, skip in zip(self.upblocks, reversed(skips)):
            # print('up', x.shape, skip.shape)
            x = l(x, skip)

        logits = self.output(x)
        return F.logsigmoid(logits), F.logsigmoid(-logits)  # log(sigmoid(logits)), log(1-sigmoid(logits))


class MONet(OOLBase):
    shortname = 'monet'
    def __init__(self,
                 n_slots=8,
                 numb=5,
                 shape=(1, 50, 50),
                 z_dim=16,
                 bg_scl = 0.09,
                 fg_scl = 0.11
                 ):
        super(MONet, self).__init__(
            pres_dist_name='unused',
            output_dist='unused',
            output_hparam=1.0,

            n_particles=1,
            z_pres_prior_p=1,
            z_where_prior_loc=[0, 0, 0, 0],
            z_where_prior_scale=[1, 1, 1, 1],
            z_what_prior_loc=[0.] * z_dim,
            z_what_prior_scale=[1.] * z_dim,
            z_depth_prior_loc=0.,
            z_depth_prior_scale=1.
        )
        self.n = n_slots
        self.shape = shape
        self.enc = ComponentEncoder(shape, z_dim)
        self.dec = ComponentDecoder(z_dim, shape)
        self.att = Attention(numb, shape)

        self.beta = .5
        self.gamma = .5
        self.bg_scl = bg_scl
        self.fg_scl = fg_scl

    def forward(self, x):
        self.training = True
        n, c, h, w = x.shape
        m = torch.zeros(n, self.n, 1, h, w, device=x.device, dtype=x.dtype)
        x_til = torch.zeros(n, self.n, c, h, w, device=x.device, dtype=x.dtype)
        m_til_logits = torch.zeros(n, self.n, 1, h, w, device=x.device, dtype=x.dtype)
        log_s_k = torch.zeros(n, 1, h, w, device=x.device, dtype=x.dtype)

        if self.training:
            kl = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
            rec_loss = torch.zeros(n, self.n, c, h, w, device=x.device, dtype=x.dtype)

        for k in range(self.n):
            if k == self.n-1:
                log_m_k = log_s_k
            else:
                log_alpha_k, log_one_minus_alpha_k = self.att(x, log_s_k)
                # print('log_alpha', log_alpha_k.shape)
                log_m_k = log_s_k + log_alpha_k
                log_s_k = log_s_k + log_one_minus_alpha_k

            loc, scale = self.enc(x, log_m_k)
            z_k_post = dist.Normal(loc, scale)
            z_k = z_k_post.rsample()

            x_til[:, k], m_til_logits[:, k] = self.dec(z_k)
            # print('xt mt', x_til[:, k].shape, m_til_logits[:, k].shape)
            m[:, k] = log_m_k.exp()

            if self.training:
                scl = self.bg_scl if k==0 else self.fg_scl
                rec_loss[:, k] = log_m_k + dist.Normal(x_til[:, k], scl).log_prob(x)
                # print('rec loss', rec_loss[:, k].shape)
                kl += kl_divergence(z_k_post, self.what_prior(z_k_post.batch_shape)).sum(1, keepdims=True) * self.beta

        m_rec = x_til * m
        canvas = m_rec.sum(1)
        m_til = torch.log_softmax(m_til_logits, dim=1)
        # print('m_til', m_til.shape)
        # print('m_rec', m_rec.shape)
        r = {
            'canvas': canvas,
            'layers': {
                'patch': x_til,
                'mask': m,
                'other_mask': m_til
            }
        }
        if self.training:
            kl_mask = F.kl_div(m_til, m, reduction='none').sum((1, 2, 3, 4)) * self.gamma
            # print('mask', kl_mask.shape)
            rec_loss = torch.logsumexp(rec_loss, dim=1).sum((1,2,3))
            # print('rec', rec_loss.shape)
            loss = -rec_loss + kl.squeeze(1) + kl_mask
            loss = loss.mean()
            r['loss'] = loss
            r['rec_loss'] = (-rec_loss).mean()
            r['kl'] = kl.mean()
            r['kl_mask'] = kl_mask.mean()
        return r

    def param_groups(self):
        return [{'params': self.parameters(), 'lr': 1}]
