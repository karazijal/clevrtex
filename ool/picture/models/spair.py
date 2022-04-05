"""
Re-implementation (heavily adjusted) of SPAIR from

"Spatially Invariant Unsupervised Object Detection with Convolutional Neural Networks"
Eric Crawford and Joelle Pineau
AAAI 2019
http://e2crawfo.github.io/pdfs/spair_aaai_2019.pdf
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from ool.picture.components.stn import STN
from ool.picture.components.mlp import MLP
from ool.picture.components.broadcast_vae import BroadcastDecoder, ConvEncoder
from .ool_base import OOLBase

SCL_EPS = 1e-21


class Backbone(nn.Module):
    """
    The paper seems to to target the cell_width of 12x12 pixels.
    """

    def __init__(self, inc):
        super(Backbone, self).__init__()
        # Suplimentary states 2 - 2 - 3 striding; code seems to use 3 - 2 - 2 striding -- probs doesn't matter
        self.conv0 = nn.Conv2d(inc, 128, kernel_size=4, stride=3)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        # Paper says 100; code seems to have 128, leaving 100 here
        self.conv5 = nn.Conv2d(128, 100, kernel_size=1, stride=1)
        self.pad2 = nn.ConstantPad2d((0, 2, 0, 2), 0)
        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)

        self.out_f = 100
        self.cell_size = 12

    def forward(self, x):
        x = self.pad1(x)
        x = F.relu(self.conv0(x), inplace=True)
        x = self.pad2(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pad2(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        return x


class LargeBackbone(nn.Sequential):
    def __init__(self, inc, extra=False):
        super(LargeBackbone, self).__init__(
            nn.Conv2d(
                inc, 16, 3, 2, 1
            ),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(
                16, 32, 3, 2, 1
            ),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(
                32, 64, 3, 2, 1
            ),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(
                128, 256, 3, 1 + int(extra), 1
            ),  # This has been changed from stride 1 to stride 2 for downsampling
            nn.CELU(),
            nn.GroupNorm(32, 256),
            nn.Conv2d(256, 100, 1),
            nn.CELU(),
            nn.GroupNorm(10, 100),
        )


class State:
    z_what_loc = None
    z_what_scl = None
    z_what = None

    kl_what = None

    z_where_loc = None
    z_where_scl = None
    z_where = None  # Actual sample
    where = None  # tranformed based on slot position
    bbox = None

    kl_where = None

    z_depth_loc = None
    z_depth_scl = None
    z_depth = None

    kl_depth = None

    z_pres_prob = None
    z_pres = None

    p_z = None
    kl_pres = None
    z_pres_logp = None
    baseline = None

    patch = None
    mask = None

    @property
    def tensor(self):
        return torch.cat([self.z_pres, self.z_where, self.z_depth, self.z_what], dim=-1)


class SlotStates:
    z_bg = None
    kl_bg = None

    @staticmethod
    def context_len(D=1, context_grid_size=1):
        return ((context_grid_size * 2 + 1) ** 2 // 2 + 1) * D

    def __init__(self, H, W, D=1, context_grid_size=1):
        self.states = np.empty((H, W, D), dtype=np.object)
        self.cgs = context_grid_size

    @property
    def tensor(self):
        n = self.states[0, 0, 0].z_what.size(0)
        H, W, D = self.states.shape

        return torch.stack(
            [
                torch.cat([self.states[h, w, d].tensor for d in range(D)], dim=-1)
                for h, w in itertools.product(range(H), range(W))
            ],
            dim=-1,
        ).view(n, -1, H, W)

    def context(self, h, w, d):
        """Return """
        context = []
        for offset_r, offset_c in itertools.product(
            range(-self.cgs, self.cgs + 1), range(-self.cgs, self.cgs + 1)
        ):
            if offset_r == 0 and offset_c == 0:
                break
            y = offset_r + h
            x = offset_c + w
            for z in range(self.states.shape[2]):
                if 0 <= y < self.states.shape[0] and 0 <= x < self.states.shape[1]:
                    context.append(self.states[y, x, z].tensor)
                else:
                    context.append(None)
        for z in range(self.states.shape[2]):
            if z < d:
                context.append(self.states[h, w, z])
            else:
                context.append(None)
        return context

    def iterslots(self):
        yield from itertools.product(
            range(self.states.shape[0]),
            range(self.states.shape[1]),
            range(self.states.shape[2]),
        )

    def iterstates(self):
        for h, w, d in self.iterslots():
            yield self.states[h, w, d]

    def update(self, h, w, d, state):
        self.states[h, w, d] = state

    def __getitem__(self, item):
        """Return tensors from State object stacked against the 1 dim"""
        return torch.stack([getattr(s, item) for s in self.iterstates()], dim=1)

    def __contains__(self, item):
        return hasattr(self.states[0, 0, 0], item)


class CombinedPred(MLP):
    def __init__(self, in_f):
        super(CombinedPred, self).__init__(
            in_f, out_f=8 + 2 + 1, h=[256, 128, 128, 64, 64, 32, 32], passf=0
        )

    def forward(self, x):
        x, passf = super(CombinedPred, self).forward(x)
        return (
            x[:, :4],
            F.softplus(x[:, 4:8]),
            x[:, 8:9],
            F.softplus(x[:, 9:10]),
            torch.sigmoid(x[:, 10:]),
        )


class BoxNetwork(MLP):
    def __init__(self, in_f=1, passf=0):
        super(BoxNetwork, self).__init__(in_f, out_f=8, h=[100, 100], passf=passf)

    def forward(self, x):
        x, passf = super(BoxNetwork, self).forward(x)
        return x[:, :4], F.softplus(x[:, 4:]), passf  # w,h,x,y x2 for mean and std


class DepthNetwork(MLP):
    def __init__(self, in_f=1, passf=0):
        super(DepthNetwork, self).__init__(in_f, out_f=2, h=[100, 100], passf=passf)

    def forward(self, x):
        x, passf = super(DepthNetwork, self).forward(x)
        return x[:, :1], F.softplus(x[:, 1:]), passf  # y, x, h, w x2 for mean and std


class PresNetwork(MLP):
    def __init__(self, in_f=1, passf=0):
        super(PresNetwork, self).__init__(in_f, out_f=1, h=[100, 100], passf=passf)

    def forward(self, x):
        x, passf = super(PresNetwork, self).forward(x)
        return torch.sigmoid(x), passf


class ObjEncoder(MLP):
    def __init__(self, in_f=1, what_dim=50):
        if isinstance(in_f, (tuple, list)):
            acc = 1.0
            for inf in in_f:
                acc *= inf
            in_f = acc
        super(ObjEncoder, self).__init__(in_f, what_dim * 2, h=[256, 128])
        self.d = what_dim

    def forward(self, x):
        x, _ = super(ObjEncoder, self).forward(x)
        return x[:, : self.d], F.softplus(x[:, self.d :])


class ObjDecoder(MLP):
    def __init__(
        self,
        obj_shape=(14, 14),
        what_dim=50,
        RGBA_scl=(1.0, 1.0, 1.0, 1.0),
        RGBA_bias=(0.0, 0.0, 0.0, 0.0),
        nch=3,
    ):
        if len(obj_shape) == 3:
            obj_shape = obj_shape[1:]
        super(ObjDecoder, self).__init__(
            what_dim, np.product(obj_shape) * (nch + 1), h=[128, 256]
        )
        self.d = obj_shape
        self.RGBA_scl = torch.tensor(RGBA_scl[:nch] + RGBA_scl[-1:]).view(1, -1, 1, 1)
        self.RGBA_bias = torch.tensor(RGBA_bias[:nch] + RGBA_bias[-1:]).view(
            1, -1, 1, 1
        )
        self.nch = nch

    def forward(self, x):
        x, _ = super(ObjDecoder, self).forward(x)
        x = x.view(x.size(0), self.nch + 1, *self.d)
        x = x * self.RGBA_scl.to(x.dtype).to(x.device) + self.RGBA_bias.to(x.dtype).to(
            x.device
        )
        return torch.sigmoid(x)


class SlotPrior:
    def __init__(self, n, h, w, d, z_pres_prior_p):
        self.HWD = h * w * d
        self.pres_prior_support = torch.arange(
            0, self.HWD + 1, device=z_pres_prior_p.device, dtype=z_pres_prior_p.dtype
        )
        pres_prior = (1 - z_pres_prior_p) * (z_pres_prior_p ** self.pres_prior_support)
        self.pres_prior = (pres_prior / pres_prior.sum()).repeat(
            n, 1
        )  # Normalise to have support over [0, HWD]
        self.counts = torch.zeros(
            n, 1, device=z_pres_prior_p.device, dtype=z_pres_prior_p.dtype
        )
        self.eps = 1e-6

    def probs(self, i):
        p_z_given_Cz = (self.pres_prior_support[None, :] - self.counts).clamp(min=0) / (
            self.HWD - i
        )
        if torch.any(torch.isnan(p_z_given_Cz)):
            print("NaN p_z_given_Cz", i)
        p_z = (self.pres_prior[:, None, :] @ p_z_given_Cz[:, :, None])[:, :, 0].clamp(
            min=self.eps, max=1.0 - self.eps
        )
        return p_z

    def update(self, z_pres, i):
        p_z_given_Cz = (self.pres_prior_support[None, :] - self.counts).clamp(min=0) / (
            self.HWD - i
        )
        z_pres = torch.round(z_pres)
        self.pres_prior = self.pres_prior * (
            z_pres * p_z_given_Cz + (1 - z_pres) * (1 - p_z_given_Cz)
        )
        self.pres_prior = self.pres_prior / self.pres_prior.sum(
            dim=1, keepdims=True
        ).clamp(min=self.eps, max=1e6)
        self.pres_prior = self.pres_prior.detach()
        self.counts += z_pres


class SPAIR(OOLBase):
    shortname = "spair"

    def local_box(self, z_where, w, h, img_shape):
        # map to 0-1 (actually -.5 - 1.5 since we want sigmoid of a distribution with unit normal prior)
        # self.min_zw = self.ensure_correct_tensor(self.min_zw)
        # self.max_zw = self.ensure_correct_tensor(self.max_zw)
        z_where = torch.sigmoid(z_where.clamp(-10, 10))
        box = (self.max_zw - self.min_zw) * z_where + self.min_zw
        wh, xy = box[:, :2], box[:, 2:]
        _wh = torch.tensor([w, h], device=box.device, dtype=box.dtype).view(1, -1)
        xy = (xy + _wh) * self.pixels_per_cell / img_shape  # in 0-1 image coords
        xy = 2 * xy - 1  # in -1 - 1 coords
        wh = wh * self.anchor / img_shape
        where = torch.hstack([wh, xy])  # w h x y
        return box, where

    def __init__(
        self,
        pixels_per_cell=12,
        n_anchors=1,
        anchor=24,
        context_grid_size=1,
        patch_size=(1, 14, 14),
        passf=100,
        what_dim=50,
        train_empty=True,
        pres_kl="original",
        factor_loss=False,
        rein=True,
        bg_dim=0,  # 0 = black bg, >0 means bg prediction
        large=False,
        std=0.15,
    ):
        super(SPAIR, self).__init__(
            pres_dist_name="relaxedbernoulli",
            z_pres_temperature=1.0,
            output_dist="normal",
            output_hparam=std,
            n_particles=10,
            z_pres_prior_p=0.99,
            z_where_prior_loc=[-2.197, -2.197, 0, 0],
            z_where_prior_scale=[0.5, 0.5, 1, 1],
            z_what_prior_loc=[0.0] * what_dim,
            z_what_prior_scale=[1.0] * what_dim,
            z_depth_prior_loc=0.0,
            z_depth_prior_scale=1.0,
            z_bg_prior_loc=[0.0] * max(bg_dim, 1),
            z_bg_prior_scale=[3.0] * max(bg_dim, 1),
        )
        self.patch_size = patch_size
        self.d = n_anchors
        self.context_grid_size = context_grid_size
        self.loss_mode = factor_loss
        self.n_passthrough_features = passf
        self.train_reinforce_override = rein
        self.pres_kl = pres_kl
        self.mask = 0.0
        self.alpha_limit = 0.0
        self.anchor = anchor
        self.pixels_per_cell = pixels_per_cell  # Downsample by factor 2 x 2 x 3
        max_zw = torch.tensor([1.0, 1.0, 1.5, 1.5])
        min_zw = torch.tensor([0.0, 0.0, -0.5, -0.5])
        self.register_buffer("max_zw", max_zw)
        self.register_buffer("min_zw", min_zw)

        empty = torch.zeros((1 + 4 + 1 + what_dim))
        if train_empty:
            self.register_parameter("empty_element", nn.Parameter(empty))
        else:
            self.register_buffer("empty_element", empty)

        self.beta_where = 1.0
        self.beta_what = 1.0
        self.beta_depth = 1.0
        self.beta_pres = 1.0
        self.beta_bg = 1.0
        if large:
            self.backbone = LargeBackbone(patch_size[0], True)
            self.pixels_per_cell = 2 ** 5
        else:
            self.backbone = Backbone(patch_size[0])
        self.stn = STN()
        self.baseline = MLP(
            100
            + SlotStates.context_len(self.d, self.context_grid_size) * (6 + what_dim),
            1,
        )
        print(SlotStates.context_len(self.d, self.context_grid_size))

        self.box_enc = BoxNetwork(
            100
            + SlotStates.context_len(self.d, self.context_grid_size) * (6 + what_dim),
            passf=self.n_passthrough_features,
        )

        self.dep_enc = DepthNetwork(
            100
            + SlotStates.context_len(self.d, self.context_grid_size) * (6 + what_dim)
            + 4
            + what_dim
            + self.n_passthrough_features,
            passf=self.n_passthrough_features,
        )
        self.pre_enc = PresNetwork(
            100
            + SlotStates.context_len(self.d, self.context_grid_size) * (6 + what_dim)
            + 4
            + what_dim
            + self.n_passthrough_features
            + 1
        )

        # This might actually be just patch...
        self.obj_enc = ObjEncoder(100 + np.product(patch_size) + 4, what_dim=what_dim)
        # self.obj_dec = ObjDecoder(patch_size, what_dim, nch=patch_size[0])
        self._obj_dec = BroadcastDecoder(z_dim=what_dim, out_ch=patch_size[0] + 1)
        self.obj_dec = lambda x: self._obj_dec(x, patch_size)
        self.obj_dec.nch = patch_size[0]
        self.bg_dim = bg_dim
        if self.bg_dim > 0:
            self.bg_enc = ConvEncoder(patch_size[0], bg_dim)
            # self.bg_enc = BgEncoder(100 + 6 + what_dim, bg_dim)
            # self.bg_enc = BgEncoder(100, bg_dim)
            # self.bg_enc = BgEncoder(patch_size[0] + 6 + what_dim, bg_dim)
            # self.bg_backbone = Backbone(patch_size[0])
            self.bg_dec = BroadcastDecoder(bg_dim, patch_size[0])
            # self.bg_enc = ObjEncoder(3*64*64, bg_dim)
            # self.bg_dec = ObjDecoder((64, 64), bg_dim, nch=2)

    @property
    def _use_reinforce(self):
        return self.pres_dist == "bernoulli" or self.train_reinforce_override

    @property
    def _use_baseline(self):
        return self.baseline is not None

    def step(self, i, h, w, d, wh_shape, data, enc_data, prior, states):
        state = State()
        n = enc_data.size(0)
        base_feat = enc_data[:, :, h, w]
        context = [
            self.empty_element.repeat(n, 1) if c is None else c
            for c in states.context(h, w, d)
        ]
        # Where
        box_input = torch.hstack([base_feat] + context)
        state.z_where_loc, state.z_where_scl, passf = self.box_enc(box_input)
        z_where_post = dist.Normal(
            state.z_where_loc, state.z_where_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_where = z_where_post.rsample()
        box, state.where = self.local_box(state.z_where, w, h, wh_shape)
        state.bbox = self.stn.bbox(state.where, wh_shape.squeeze())

        # Look
        state.glimpse = self.stn.extract(data, state.where, patch_size=self.patch_size)

        # What
        obj_input = torch.hstack([base_feat, box, state.glimpse.view(n, -1)])
        state.z_what_loc, state.z_what_scl = self.obj_enc(obj_input)
        z_what_post = dist.Normal(
            state.z_what_loc, state.z_what_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_what = z_what_post.rsample()

        # Depth
        dep_input = torch.hstack([base_feat, *context, box, state.z_what, passf])
        state.z_depth_loc, state.z_depth_scl, passf = self.dep_enc(dep_input)
        z_depth_post = dist.Normal(
            state.z_depth_loc, state.z_depth_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_depth = torch.sigmoid(z_depth_post.rsample().clamp(-10, 10))

        # Pres
        pre_input = torch.hstack(
            [base_feat, *context, box, state.z_what, state.z_depth, passf]
        )
        state.z_pres_prob, _ = self.pre_enc(pre_input)

        if torch.any(torch.isnan(state.z_pres_prob)):
            print("NaN qz_pres", i)
            # import ipdb; ipdb.set_trace()

        if torch.any(torch.isinf(state.z_pres_prob)):
            print("INF qz_pres", i)
            # import ipdb; ipdb.set_trace()

        z_pres_post = self.pres_dist(state.z_pres_prob)
        state.z_pres = z_pres_post.sample()
        state.z_pres_logp = z_pres_post.log_prob(state.z_pres)

        # Reconstruct
        robj = self.obj_dec(state.z_what)
        state.robj = robj[:, : self.obj_dec.nch]
        state.rmsk = robj[:, -1:] * state.z_pres.view(-1, 1, 1, 1)
        r_img = self.stn.place(
            state.where, robj, (self.obj_dec.nch + 1, *data.shape[-2:])
        )
        state.patch = r_img[:, : self.obj_dec.nch]
        if robj.size(1) - 1 == 1:
            state.patch = torch.ones_like(state.patch)
        state.mask = r_img[:, -1:] * state.z_pres.view(-1, 1, 1, 1)

        if self.training:
            state.kl_where = (
                kl_divergence(
                    z_where_post, self.where_prior(z_where_post.batch_shape)
                ).sum(1)
                * self.beta_where
            )
            state.kl_what = (
                kl_divergence(
                    z_what_post, self.what_prior(z_what_post.batch_shape)
                ).sum(1)
                * self.beta_what
            )
            state.kl_depth = (
                kl_divergence(
                    z_depth_post, self.depth_prior(z_depth_post.batch_shape)
                ).sum(1)
                * self.beta_depth
            )
            if torch.any(torch.isinf(state.kl_where)):
                pass
                # import ipdb; ipdb.set_trace()
            p_z = prior.probs(i)
            prior.update(state.z_pres, i)
            if self.pres_kl == "original":
                state.kl_pres = (
                    z_pres_post.probs * (z_pres_post.probs.log() - p_z.log())
                    + (1 - z_pres_post.probs)
                    * ((1 - z_pres_post.probs).log() - (1 - p_z).log())
                ).sum(1) * self.beta_pres
                if torch.any(torch.isinf(state.kl_pres)):
                    pass
                    # import ipdb; ipdb.set_trace()
            elif self.pres_kl == "match":
                prior_dist = self.pres_dist(p_z, batch_shape=z_pres_post.batch_shape)
                state.kl_pres = (
                    kl_divergence(z_pres_post, prior_dist).sum(1) * self.beta_pres
                )
            elif self.pres_kl == "ind":
                prior_dist = self.pres_dist(
                    self.z_pres_prior_p.repeat(n, 1), z_pres_post.batch_shape
                )
                state.kl_pres = (
                    kl_divergence(z_pres_post, prior_dist).sum(1) * self.beta_pres
                )

            if self.mask:
                state.kl_where = (
                    (1 - self.mask) * state.kl_where
                    + self.mask * state.kl_where * state.z_pres.squeeze()
                )
                state.kl_what = (
                    (1 - self.mask) * state.kl_what
                    + self.mask * state.kl_what * state.z_pres.squeeze()
                )
                state.kl_depth = (
                    (1 - self.mask) * state.kl_depth
                    + self.mask * state.kl_depth * state.z_pres.squeeze()
                )

            if self._use_baseline:
                state.baseline, _ = self.baseline(box_input.detach())
                state.baseline = state.baseline.view(-1, 1)
                # if self.mask_pres:
                # state.baseline = state.baseline * state.z_pres.detach()

        return state

    def build_bg(self, data, bg_inp):
        kl_bg = torch.zeros(data.size(0), device=data.device, dtype=data.dtype)
        z_bg = None
        if self.bg_dim:
            # bg_data = self.bg_backbone(data)
            # bg_inp = torch.cat([bg_data, bg_inp], dim=1)
            bg_inp = data
            bg_loc, bg_scale = self.bg_enc(bg_inp)
            # background = torch.ones_like(data) * torch.sigmoid(bg_loc)[:, :, None, None]
            z_bg_post = dist.Normal(bg_loc, bg_scale.clamp(SCL_EPS, 1e19))
            z_bg = z_bg_post.rsample()

            if self.training:
                kl_bg = (
                    kl_divergence(z_bg_post, self.bg_prior(z_bg_post.batch_shape)).sum(
                        1
                    )
                    * self.beta_bg
                )
                if torch.any(torch.isnan(kl_bg)):
                    print("kl_bg nan")
                if torch.any(torch.isinf(kl_bg)):
                    print("kl_bg inf")

            background = self.bg_dec(z_bg, data.shape)
            # c, w, h = self.patch_size[-3:]
            # background = self.bg_dec(z_bg, (c, 2*h, 2*w))

            if background.shape != data.shape:
                background = F.interpolate(
                    background, size=data.shape[-2:], mode="nearest", align_corners=None
                )

            if torch.any(torch.isnan(background)):
                print("bg nan")
            if torch.any(torch.isinf(background)):
                print("bg inf")

            return background, kl_bg, z_bg
        return torch.zeros_like(data), kl_bg, z_bg

    def forward(self, data, return_state=True):
        enc_x = self.backbone(data)
        n, c, H, W = enc_x.shape
        self.onceprint(f"Encoded shape: {(c, H, W)}")
        prior = SlotPrior(n, H, W, self.d, self.z_pres_prior_p)
        states = SlotStates(H, W, self.d, self.context_grid_size)
        img_shape = torch.tensor(
            [data.size(-2), data.size(-1)], device=data.device, dtype=data.dtype
        ).view(1, -1)
        wh_shape = torch.tensor(
            [data.size(3), data.size(2)], device=data.device, dtype=data.dtype
        ).view(1, -1)
        heatmap = torch.zeros(n, 1, H, W, device=data.device, dtype=data.dtype)
        for i, (h, w, d) in enumerate(states.iterslots()):
            state = self.step(i, h, w, d, wh_shape, data, enc_x, prior, states)
            states.update(h, w, d, state)
            heatmap[:, :, h, w] = state.z_pres_prob

        background, states.kl_bg, states.z_bg = self.build_bg(
            data, bg_inp=states.tensor
        )
        if self.loss_mode:
            canvas, blended_masks = self.render2(
                states["z_depth"], states["patch"], states["mask"], background
            )
        else:
            canvas, blended_masks = self.render(
                states["z_depth"], states["patch"], states["mask"], background
            )
        r = {
            "canvas_loc": canvas,
            "background": background,
            "reinforce_target": None,
            "baseline": None,
            "mask": None,
            "counts": torch.round(states["z_pres"]).sum((1, 2)),
            "shape": (H, W),
            "heatmap": heatmap,
            "steps": states,  # This acts as a dict
        }

        if self.training:
            r = self.build_loss(data, r, states, blended_masks)
        else:
            r["canvas"] = self.output_dist(canvas).mean
        return r

    def render(self, z_depth_slots, objects, masks, background=None):
        depth = z_depth_slots[:, :, :, None, None]
        if torch.any(depth == 0):
            print("z_depth == 0")
        imprt = torch.maximum(
            masks / depth, torch.tensor(0.01, dtype=depth.dtype, device=depth.device)
        )

        background = background[:, None]  # New dim for n_obj

        blended_objts = (objects * masks + background * (1 - masks)) * imprt

        canvas = blended_objts.sum(dim=1) / imprt.sum(dim=1).clamp(min=1e-6)
        if torch.any(canvas < 0) or torch.any(canvas > 1):
            print("canvas oob")
        return canvas, None

    def render2(self, z_depth_slots, objects, masks, background=None):
        objts = objects * masks
        imprt = 100 * z_depth_slots[:, :, :, None, None] * masks
        imprt = torch.softmax(imprt, dim=1)
        blended_masks = (masks * imprt).sum(1)
        blended_masks = blended_masks.clamp(min=self.alpha_limit)
        blended_objects = (objts * imprt).sum(1)
        canvas = blended_masks * blended_objects + (1 - blended_masks) * background
        return canvas, blended_masks

    def build_loss(self, data, r, states, blended_masks=None):
        baseline_loss = torch.tensor(0.0)
        reinforce = torch.tensor(0.0)
        output_dist = self.output_dist(r["canvas_loc"])
        r["canvas"] = output_dist.mean
        if blended_masks is None:
            recon_loss = output_dist.log_prob(data).sum((1, 2, 3))
        else:
            rec = output_dist.log_prob(data)
            fg_loss = rec + blended_masks.clamp(1e-6).log()
            bg_loss = rec + (1.0 - blended_masks).clamp(1e-6).log()
            recon_loss = torch.logsumexp(
                torch.stack([fg_loss, bg_loss], dim=1), dim=1
            ).sum((1, 2, 3))

        kl = (
            states["kl_where"]
            + states["kl_what"]
            + states["kl_depth"]
            + states["kl_pres"]
        )
        if self._use_reinforce:
            rt = torch.flip(torch.cumsum(torch.flip(kl, dims=(1,)), dim=1), dims=(1,))
            mult = recon_loss[:, None] - states.kl_bg[:, None]
            if self.mask:
                mult = mult * states["z_pres"].squeeze(2).detach()
            rt = rt - mult

        elbo = recon_loss - kl.sum(1) - states.kl_bg
        loss = -elbo
        if self._use_reinforce:
            if self._use_baseline:
                baselines = states["baseline"].squeeze(2)
                baseline_loss = F.mse_loss(baselines, rt.detach(), reduction="none")
                baseline_loss = baseline_loss.sum(1).mean(0)
                rt = rt - baselines

            reinforce = rt.detach() * states["z_pres_logp"].squeeze(2)
            reinforce = reinforce.sum(1)
            if self.pres_dist_name == "bernoulli":
                loss += reinforce

        loss = loss.mean()
        if self._use_baseline and self._use_reinforce:
            loss += baseline_loss
        r["loss"] = loss
        r["elbo"] = elbo.mean()
        r["kl"] = kl.mean()
        r["rec_loss"] = recon_loss.mean()
        r["rein_loss"] = reinforce.mean() if reinforce is not None else reinforce
        r["base_loss"] = baseline_loss
        r["kl_where"] = states["kl_where"].flatten(1).sum(-1).mean()
        r["kl_what"] = states["kl_what"].flatten(1).sum(-1).mean()
        r["kl_depth"] = states["kl_depth"].flatten(1).sum(-1).mean()
        r["kl_pres"] = states["kl_pres"].flatten(1).sum(-1).mean()
        return r


class SimpleSPAIR(SPAIR):
    shortname = "spair-s"

    def __init__(self, **kwargs):
        super(SimpleSPAIR, self).__init__(**kwargs)
        del self.box_enc
        del self.dep_enc
        del self.pre_enc
        what_dim = kwargs["what_dim"]
        self.z_enc = CombinedPred(
            100
            + SlotStates.context_len(self.d, self.context_grid_size) * (6 + what_dim)
        )

    def step(self, i, h, w, d, img_shape, data, enc_data, prior, states):
        state = State()
        n = enc_data.size(0)
        base_feat = enc_data[:, :, h, w]
        context = [
            self.empty_element.repeat(n, 1) if c is None else c
            for c in states.context(h, w, d)
        ]

        # Where
        inp = torch.hstack([base_feat] + context)
        (
            state.z_where_loc,
            state.z_where_scl,
            state.z_depth_loc,
            state.z_depth_scl,
            state.z_pres_prob,
        ) = self.z_enc(inp)

        z_where_post = dist.Normal(
            state.z_where_loc, state.z_where_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_where = z_where_post.rsample()
        box, state.where = self.local_box(state.z_where, w, h, img_shape)

        # Look
        state.glimpse = self.stn.extract(data, state.where, patch_size=self.patch_size)

        # What
        obj_input = torch.hstack([base_feat, box, state.glimpse.view(n, -1)])
        state.z_what_loc, state.z_what_scl = self.obj_enc(obj_input)
        z_what_post = dist.Normal(
            state.z_what_loc, state.z_what_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_what = z_what_post.rsample()

        # Depth
        z_depth_post = dist.Normal(
            state.z_depth_loc, state.z_depth_scl.clamp(SCL_EPS, 1e19)
        )
        state.z_depth = torch.sigmoid(z_depth_post.rsample().clamp(-10, 10))

        # Pres
        if torch.any(torch.isnan(state.z_pres_prob)):
            print("NaN qz_pres", i)
            import ipdb

            ipdb.set_trace()

        if torch.any(torch.isinf(state.z_pres_prob)):
            print("INF qz_pres", i)
            import ipdb

            ipdb.set_trace()

        z_pres_post = self.pres_dist(state.z_pres_prob)
        state.z_pres = z_pres_post.sample()
        state.z_pres_logp = z_pres_post.log_prob(state.z_pres)

        # Reconstruct
        robj = self.obj_dec(state.z_what)
        state.robj = robj[:, : self.obj_dec.nch]
        state.rmsk = robj[:, -1:] * state.z_pres.view(-1, 1, 1, 1)
        r_img = self.stn.place(
            state.where, robj, (self.obj_dec.nch + 1, *data.shape[-2:])
        )
        state.patch = r_img[:, : self.obj_dec.nch]
        if robj.size(1) - 1 == 1:
            state.patch = torch.ones_like(state.patch)
        state.mask = r_img[:, -1:] * state.z_pres.view(-1, 1, 1, 1)

        if self.training:
            state.kl_where = (
                kl_divergence(
                    z_where_post, self.where_prior(z_where_post.batch_shape)
                ).sum(1)
                * self.beta_where
            )
            state.kl_what = (
                kl_divergence(
                    z_what_post, self.what_prior(z_what_post.batch_shape)
                ).sum(1)
                * self.beta_what
            )
            state.kl_depth = (
                kl_divergence(
                    z_depth_post, self.depth_prior(z_depth_post.batch_shape)
                ).sum(1)
                * self.beta_depth
            )
            if torch.any(torch.isinf(state.kl_where)):
                import ipdb

                ipdb.set_trace()
            p_z = prior.probs(i)
            prior.update(state.z_pres, i)
            if self.pres_kl == "original":
                state.kl_pres = (
                    z_pres_post.probs * (z_pres_post.probs.log() - p_z.log())
                    + (1 - z_pres_post.probs)
                    * ((1 - z_pres_post.probs).log() - (1 - p_z).log())
                ).sum(1) * self.beta_pres
                if torch.any(torch.isinf(state.kl_pres)):
                    import ipdb

                    ipdb.set_trace()
            elif self.pres_kl == "match":
                prior_dist = self.pres_dist(p_z, batch_shape=z_pres_post.batch_shape)
                state.kl_pres = (
                    kl_divergence(z_pres_post, prior_dist).sum(1) * self.beta_pres
                )
            elif self.pres_kl == "ind":
                prior_dist = self.pres_dist(
                    self.z_pres_prior_p.repeat(n, 1), z_pres_post.batch_shape
                )
                state.kl_pres = (
                    kl_divergence(z_pres_post, prior_dist).sum(1) * self.beta_pres
                )

            if self.mask:
                state.kl_where = (
                    (1 - self.mask) * state.kl_where
                    + self.mask * state.kl_where * state.z_pres.squeeze()
                )
                state.kl_what = (
                    (1 - self.mask) * state.kl_what
                    + self.mask * state.kl_what * state.z_pres.squeeze()
                )
                state.kl_depth = (
                    (1 - self.mask) * state.kl_where
                    + self.mask * state.kl_where * state.z_pres.squeeze()
                )

            if self._use_baseline:
                state.baseline, _ = self.baseline(inp.detach())
                state.baseline = state.baseline.view(-1, 1)
                # if self.mask_pres:
                # state.baseline = state.baseline * state.z_pres.detach()

        return state


class BroadCastSPAIR(SPAIR):
    shortname = "spair-b"

    def __int__(self, **kwargs):
        super(BroadCastSPAIR, self).__int__(**kwargs)
        what_dim = kwargs["what_dim"]
        patch_size = kwargs["patch_size"]
        self._obj_dec = BroadcastDecoder(what_dim, patch_size[0] + 1)
        self.obj_dec = lambda x: self._obj_dec(x, patch_size)
