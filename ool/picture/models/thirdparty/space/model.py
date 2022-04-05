"""
Code adjusted from https://github.com/zhixuan-lin/SPACE

"SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition"
Zhixuan Lin, Yi-Fu Wu, Skand Vishwanath Peri, Weihao Sun, Gautam Singh, Fei Deng, Jindong Jiang, Sungjin Ahn
ICLR 2020
https://arxiv.org/abs/2001.02407
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, RelaxedBernoulli
from torch.distributions.utils import broadcast_all

arch = lambda: None
arch.__dict__ = {
    "img_shape": (128, 128),
    # Grid size. There will be G*G slots
    "G": 8,
    # Foreground configurations
    # ==== START ====
    # Foreground likelihood sigma
    "fg_sigma": 0.15,
    # Size of the glimpse
    "glimpse_size": 32,
    # Encoded image feature channels
    "img_enc_dim_fg": 128,
    # Latent dimensions
    "z_pres_dim": 1,
    "z_depth_dim": 1,
    # (h, w)
    "z_where_scale_dim": 2,
    # (x, y)
    "z_where_shift_dim": 2,
    "z_what_dim": 32,
    # z_pres prior
    "z_pres_start_step": 4000,
    "z_pres_end_step": 10000,
    "z_pres_start_value": 0.1,
    "z_pres_end_value": 0.01,
    # z_scale prior
    "z_scale_mean_start_step": 10000,
    "z_scale_mean_end_step": 20000,
    "z_scale_mean_start_value": -1.0,
    "z_scale_mean_end_value": -2.0,
    "z_scale_std_value": 0.1,
    # Temperature for gumbel-softmax
    "tau_start_step": 0,
    "tau_end_step": 10000,
    "tau_start_value": 2.5,
    "tau_end_value": 2.5,
    # Turn on boundary loss or not
    "boundary_loss": True,
    # When to turn off boundary loss
    "bl_off_step": 100000000,
    # Fix alpha for the first N steps
    "fix_alpha_steps": 0,
    "fix_alpha_value": 0.1,
    # ==== END ====
    # Background configurations
    # ==== START ====
    # Number of background components. If you set this to one, you should use a strong decoder instead.
    "K": 5,
    # Background likelihood sigma
    "bg_sigma": 0.15,
    # Image encoding dimension
    "img_enc_dim_bg": 64,
    # Latent dimensions
    "z_mask_dim": 32,
    "z_comp_dim": 32,
    # (H, W)
    "rnn_mask_hidden_dim": 64,
    # This should be same as above
    "rnn_mask_prior_hidden_dim": 64,
    # Hidden layer dim for the network that computes q(z_c|z_m, x)
    "predict_comp_hidden_dim": 64,
    # ==== END ====
}


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = (
        z_where[:, 2] if not inverse else -z_where[:, 2] / (z_where[:, 0] + 1e-9)
    )
    theta[:, 1, -1] = (
        z_where[:, 3] if not inverse else -z_where[:, 3] / (z_where[:, 1] + 1e-9)
    )
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)


def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing

    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if step <= start_step:
        x = torch.tensor(start_value, device=device)
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=device)
    else:
        x = torch.tensor(end_value, device=device)

    return x


class NumericalRelaxedBernoulli(RelaxedBernoulli):
    """
    This is a bit weird. In essence it is just RelaxedBernoulli with logit as input.
    """

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)

        out = self.temperature.log() + diff - 2 * diff.exp().log1p()

        return out


def kl_divergence_bern_bern(z_pres_logits, prior_pres_prob, eps=1e-15):
    """
    Compute kl divergence of two Bernoulli distributions
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = z_pres_probs * (
        torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)
    ) + (1 - z_pres_probs) * (
        torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps)
    )

    return kl


def get_boundary_kernel_new(kernel_size=32, boundary_width=6):
    """
    Will return something like this:
    ============
    =          =
    =          =
    ============
    size will be (kernel_size, kernel_size)
    """
    filter = torch.zeros(kernel_size, kernel_size)
    filter[:, :] = 1.0 / (kernel_size ** 2)
    # Set center to zero
    filter[
        boundary_width : kernel_size - boundary_width,
        boundary_width : kernel_size - boundary_width,
    ] = 0.0

    return filter


class SpaceFg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.img_encoder = ImgEncoderFg()
        self.z_what_net = ZWhatEnc()
        self.glimpse_dec = GlimpseDec()
        # This is what is really used
        self.boundary_kernel = get_boundary_kernel_new(kernel_size=32, boundary_width=6)

        self.fg_sigma = arch.fg_sigma
        # I register many things as buffer but this is not really necessary.
        # Temperature for gumbel-softmax
        self.register_buffer("tau", torch.tensor(arch.tau_start_value))

        # Priors
        self.register_buffer("prior_z_pres_prob", torch.tensor(arch.z_pres_start_value))
        self.register_buffer("prior_what_mean", torch.zeros(1))
        self.register_buffer("prior_what_std", torch.ones(1))
        self.register_buffer("prior_depth_mean", torch.zeros(1))
        self.register_buffer("prior_depth_std", torch.ones(1))
        self.prior_scale_mean_new = torch.tensor(arch.z_scale_mean_start_value)
        self.prior_scale_std_new = torch.tensor(arch.z_scale_std_value)
        self.prior_shift_mean_new = torch.tensor(0.0)
        self.prior_shift_std_new = torch.tensor(1.0)
        # self.register_buffer('prior_scale_mean_new', torch.tensor(arch.z_scale_mean_start_value))
        # self.register_buffer('prior_scale_std_new', torch.tensor(arch.z_scale_std_value))
        # self.register_buffer('prior_shift_mean_new', torch.tensor(0.))
        # self.register_buffer('prior_shift_std_new', torch.tensor(1.))

        # # TODO: These are placeholders for loading old checkpoints. No longer used
        # self.boundary_filter = get_boundary_kernel(sigma=20)
        # self.register_buffer('prior_scale_mean',
        #                      torch.tensor([arch.z_scale_mean_start_value] * 2).view((arch.z_where_scale_dim), 1, 1))
        # self.register_buffer('prior_scale_std',
        #                      torch.tensor([arch.z_scale_std_value] * 2).view((arch.z_where_scale_dim), 1, 1))
        # self.register_buffer('prior_shift_mean',
        #                      torch.tensor([0., 0.]).view((arch.z_where_shift_dim), 1, 1))
        # self.register_buffer('prior_shift_std',
        #                      torch.tensor([1., 1.]).view((arch.z_where_shift_dim), 1, 1))

    @property
    def z_what_prior(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def z_depth_prior(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def z_scale_prior(self):
        return Normal(self.prior_scale_mean_new, self.prior_scale_std_new)

    @property
    def z_shift_prior(self):
        return Normal(self.prior_shift_mean_new, self.prior_shift_std_new)

    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """

        self.prior_z_pres_prob = linear_annealing(
            self.prior_z_pres_prob.device,
            global_step,
            arch.z_pres_start_step,
            arch.z_pres_end_step,
            arch.z_pres_start_value,
            arch.z_pres_end_value,
        )
        self.prior_scale_mean_new = linear_annealing(
            self.prior_z_pres_prob.device,
            global_step,
            arch.z_scale_mean_start_step,
            arch.z_scale_mean_end_step,
            arch.z_scale_mean_start_value,
            arch.z_scale_mean_end_value,
        )
        self.tau = linear_annealing(
            self.tau.device,
            global_step,
            arch.tau_start_step,
            arch.tau_end_step,
            arch.tau_start_value,
            arch.tau_end_value,
        )

    def forward(self, x, globel_step):
        """
        Forward pass

        :param x: (B, 3, H, W)
        :param globel_step: global step (training)
        :return:
            fg_likelihood: (B, 3, H, W)
            y_nobg: (B, 3, H, W), foreground reconstruction
            alpha_map: (B, 1, H, W)
            kl: (B,) total foreground kl
            boundary_loss: (B,)
            log: a dictionary containing anything we need for visualization
        """
        B = x.size(0)
        # if globel_step:
        self.anneal(globel_step)

        # Everything is (B, G*G, D), where D varies
        (
            z_pres,
            z_depth,
            z_scale,
            z_shift,
            z_where,
            z_pres_logits,
            z_depth_post,
            z_scale_post,
            z_shift_post,
        ) = self.img_encoder(x, self.tau)

        # (B, 3, H, W) -> (B*G*G, 3, H, W). Note we must use repeat_interleave instead of repeat
        x_repeat = torch.repeat_interleave(x, arch.G ** 2, dim=0)

        # (B*G*G, 3, H, W), where G is the grid size
        # Extract glimpse
        x_att = spatial_transform(
            x_repeat,
            z_where.view(B * arch.G ** 2, 4),
            (B * arch.G ** 2, 3, arch.glimpse_size, arch.glimpse_size),
            inverse=False,
        )

        # (B*G*G, D)
        z_what, z_what_post = self.z_what_net(x_att)

        # Decode z_what into small reconstructed glimpses
        # All (B*G*G, 3, H, W)
        o_att, alpha_att = self.glimpse_dec(z_what)
        # z_pres: (B, G*G, 1) -> (B*G*G, 1, 1, 1)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        # (B*G*G, 3, H, W)
        y_att = alpha_att_hat * o_att

        # Compute pixel-wise object weights
        # (B*G*G, 1, H, W). These are glimpse size
        importance_map = (
            alpha_att_hat
            * 100.0
            * torch.sigmoid(-z_depth.view(B * arch.G ** 2, 1, 1, 1))
        )
        # (B*G*G, 1, H, W). These are of full resolution
        importance_map_full_res = spatial_transform(
            importance_map,
            z_where.view(B * arch.G ** 2, 4),
            (B * arch.G ** 2, 1, *arch.img_shape),
            inverse=True,
        )

        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        importance_map_full_res = importance_map_full_res.view(
            B, arch.G ** 2, 1, *arch.img_shape
        )
        # Normalize (B, >G*G<, 1, H, W)
        importance_map_full_res_norm = torch.softmax(importance_map_full_res, dim=1)

        # To full resolution
        # (B*G*G, 3, H, W) -> (B, G*G, 3, H, W)
        y_each_cell = spatial_transform(
            y_att,
            z_where.view(B * arch.G ** 2, 4),
            (B * arch.G ** 2, 3, *arch.img_shape),
            inverse=True,
        ).view(B, arch.G ** 2, 3, *arch.img_shape)
        # Weighted sum, (B, 3, H, W)
        y_nobg = (y_each_cell * importance_map_full_res_norm).sum(dim=1)

        # To full resolution
        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        alpha_each_cell = spatial_transform(
            alpha_att_hat,
            z_where.view(B * arch.G ** 2, 4),
            (B * arch.G ** 2, 1, *arch.img_shape),
            inverse=True,
        ).view(B, arch.G ** 2, 1, *arch.img_shape)

        # Weighted sum, (B, 1, H, W)
        alpha_map = (alpha_each_cell * importance_map_full_res_norm).sum(dim=1)

        # Everything is computed. Now let's compute loss
        # Compute KL divergences
        # (B, G*G, 1)
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)

        # (B, G*G, 1)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)

        # (B, G*G, 2)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_shift = kl_divergence(z_shift_post, self.z_shift_prior)

        # Reshape z_what and z_what_post
        # (B*G*G, D) -> (B, G*G, D)
        z_what = z_what.view(B, arch.G ** 2, arch.z_what_dim)
        z_what_post = Normal(
            *[
                x.view(B, arch.G ** 2, arch.z_what_dim)
                for x in [z_what_post.mean, z_what_post.stddev]
            ]
        )
        # (B, G*G, D)
        kl_z_what = kl_divergence(z_what_post, self.z_what_prior)

        # dimensionality check
        assert (
            (kl_z_pres.size() == (B, arch.G ** 2, 1))
            and (kl_z_depth.size() == (B, arch.G ** 2, 1))
            and (kl_z_scale.size() == (B, arch.G ** 2, 2))
            and (kl_z_shift.size() == (B, arch.G ** 2, 2))
            and (kl_z_what.size() == (B, arch.G ** 2, arch.z_what_dim))
        )

        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what = [
            x.flatten(start_dim=1).sum(1)
            for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what]
        ]
        # (B,)
        kl_z_where = kl_z_scale + kl_z_shift

        # Compute boundary loss
        # (1, 1, K, K)
        boundary_kernel = self.boundary_kernel[None, None].to(x.device)
        # (1, 1, K, K) * (B*G*G, 1, 1) -> (B*G*G, 1, K, K)
        boundary_kernel = boundary_kernel * z_pres.view(B * arch.G ** 2, 1, 1, 1)
        # (B, G*G, 1, H, W), to full resolution
        boundary_map = spatial_transform(
            boundary_kernel,
            z_where.view(B * arch.G ** 2, 4),
            (B * arch.G ** 2, 1, *arch.img_shape),
            inverse=True,
        ).view(B, arch.G ** 2, 1, *arch.img_shape)
        # (B, 1, H, W)
        boundary_map = boundary_map.sum(dim=1)
        # TODO: some magic number. For reproducibility I will keep it
        boundary_map = boundary_map * 1000
        # (B, 1, H, W) * (B, 1, H, W)
        overlap = boundary_map * alpha_map
        # TODO: another magic number. For reproducibility I will keep it
        p_boundary = Normal(0, 0.7)
        # (B, 1, H, W)
        boundary_loss = p_boundary.log_prob(overlap)
        # (B,)
        boundary_loss = boundary_loss.flatten(start_dim=1).sum(1)

        # NOTE: we want to minimize this
        boundary_loss = -boundary_loss

        # Compute foreground likelhood
        fg_dist = Normal(y_nobg, self.fg_sigma)
        fg_likelihood = fg_dist.log_prob(x)

        kl = kl_z_what + kl_z_where + kl_z_pres + kl_z_depth

        if not arch.boundary_loss or globel_step > arch.bl_off_step:
            boundary_loss = boundary_loss * 0.0

        # For visualizating
        # Dimensionality check
        assert (
            (z_pres.size() == (B, arch.G ** 2, 1))
            and (z_depth.size() == (B, arch.G ** 2, 1))
            and (z_scale.size() == (B, arch.G ** 2, 2))
            and (z_shift.size() == (B, arch.G ** 2, 2))
            and (z_where.size() == (B, arch.G ** 2, 4))
            and (z_what.size() == (B, arch.G ** 2, arch.z_what_dim))
        )
        log = {
            "fg": y_nobg,
            "z_what": z_what,
            "z_where": z_where,
            "z_pres": z_pres,
            "z_scale": z_scale,
            "z_shift": z_shift,
            "z_depth": z_depth,
            "z_pres_prob": torch.sigmoid(z_pres_logits),
            "prior_z_pres_prob": self.prior_z_pres_prob.unsqueeze(0),
            "o_att": o_att,
            "alpha_att_hat": alpha_att_hat,
            "alpha_att": alpha_att,
            "alpha_map": alpha_map,
            "boundary_loss": boundary_loss,
            "boundary_map": boundary_map,
            "importance_map_full_res_norm": importance_map_full_res_norm,
            "kl_z_what": kl_z_what,
            "kl_z_pres": kl_z_pres,
            "kl_z_scale": kl_z_scale,
            "kl_z_shift": kl_z_shift,
            "kl_z_depth": kl_z_depth,
            "kl_z_where": kl_z_where,
            "patches": y_each_cell,
            "patch_masks": alpha_each_cell,
        }
        return fg_likelihood, y_nobg, alpha_map, kl, boundary_loss, log


class ImgEncoderFg(nn.Module):
    """
    Foreground image encoder.
    """

    def __init__(self):
        super(ImgEncoderFg, self).__init__()

        assert arch.G in [4, 8, 16]
        # Adjust stride such that the output dimension of the volume matches (G, G, ...)
        last_stride = 2 if arch.G in [8, 4] else 1
        second_to_last_stride = 2 if arch.G in [4] else 1

        # Foreground Image Encoder in the paper
        # Encoder: (B, C, Himg, Wimg) -> (B, E, G, G)
        # G is H=W in the paper
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, second_to_last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 256, 3, last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(32, 256),
            nn.Conv2d(256, arch.img_enc_dim_fg, 1),
            nn.CELU(),
            nn.GroupNorm(16, arch.img_enc_dim_fg),
        )

        # Residual Connection in the paper
        # Lateral connection (B, E, G, G) -> (B, E, G, G)
        self.enc_lat = nn.Sequential(
            nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, arch.img_enc_dim_fg),
            nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, arch.img_enc_dim_fg),
        )

        # Residual Encoder in the paper
        # enc + lateral -> enc (B, 2*E, G, G) -> (B, 128, G, G)
        self.enc_cat = nn.Sequential(
            nn.Conv2d(arch.img_enc_dim_fg * 2, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
        )

        # Image encoding -> latent distribution parameters (B, 128, G, G) -> (B, D, G, G)
        self.z_scale_net = nn.Conv2d(128, (arch.z_where_scale_dim) * 2, 1)
        self.z_shift_net = nn.Conv2d(128, (arch.z_where_shift_dim) * 2, 1)
        self.z_pres_net = nn.Conv2d(128, arch.z_pres_dim, 1)
        self.z_depth_net = nn.Conv2d(128, arch.z_depth_dim * 2, 1)

        # (G, G). Grid center offset. (offset_x[i, j], offset_y[i, j]) is the center for cell (i, j)
        offset_y, offset_x = torch.meshgrid(
            [torch.arange(arch.G), torch.arange(arch.G)]
        )

        # (2, G, G). I do this just to ensure that device is correct.
        self.register_buffer("offset", torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self, x, tau):
        """
        Given image, infer z_pres, z_depth, z_where

        :param x: (B, 3, H, W)
        :param tau: temperature for the relaxed bernoulli
        :return
            z_pres: (B, G*G, 1)
            z_depth: (B, G*G, 1)
            z_scale: (B, G*G, 2)
            z_shift: (B, G*G, 2)
            z_where: (B, G*G, 4)
            z_pres_logits: (B, G*G, 1)
            z_depth_post: Normal, (B, G*G, 1)
            z_scale_post: Normal, (B, G*G, 2)
            z_shift_post: Normal, (B, G*G, 2)
        """
        B = x.size(0)

        # (B, C, H, W)
        img_enc = self.enc(x)
        # (B, E, G, G)
        lateral_enc = self.enc_lat(img_enc)
        # (B, 2E, G, G) -> (B, 128, H, W)
        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))

        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                B, D, G, G = x.size()
                y = x.permute(0, 2, 3, 1).view(B, G * G, D)
                out.append(y)
            return out[0] if len(args) == 1 else out

        # Compute posteriors

        # (B, 1, G, G)
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(cat_enc))
        # (B, 1, G, G) - > (B, G*G, 1)
        z_pres_logits = reshape(z_pres_logits)
        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        # Unbounded
        z_pres_y = z_pres_post.rsample()
        # in (0, 1)
        z_pres = torch.sigmoid(z_pres_y)

        # (B, 1, G, G)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        # (B, 1, G, G) -> (B, G*G, 1)
        z_depth_mean, z_depth_std = reshape(z_depth_mean, z_depth_std)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        # (B, G*G, 1)
        z_depth = z_depth_post.rsample()

        # (B, 2, G, G)
        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(cat_enc).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        # (B, 2, G, G) -> (B, G*G, 2)
        z_scale_mean, z_scale_std = reshape(z_scale_mean, z_scale_std)
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()

        # (B, 2, G, G)
        z_shift_mean, z_shift_std = self.z_shift_net(cat_enc).chunk(2, 1)
        z_shift_std = F.softplus(z_shift_std)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_shift_mean, z_shift_std = reshape(z_shift_mean, z_shift_std)
        z_shift_post = Normal(z_shift_mean, z_shift_std)
        z_shift = z_shift_post.rsample()

        # scale: unbounded to (0, 1), (B, G*G, 2)
        z_scale = z_scale.sigmoid()
        # offset: (2, G, G) -> (G*G, 2)
        offset = self.offset.permute(1, 2, 0).view(arch.G ** 2, 2)
        # (B, G*G, 2) and (G*G, 2)
        # where: (-1, 1)(local) -> add center points -> (0, 2) -> (-1, 1)
        z_shift = (2.0 / arch.G) * (offset + 0.5 + z_shift.tanh()) - 1

        # (B, G*G, 4)
        z_where = torch.cat((z_scale, z_shift), dim=-1)

        # Check dimensions
        assert (
            (z_pres.size() == (B, arch.G ** 2, 1))
            and (z_depth.size() == (B, arch.G ** 2, 1))
            and (z_shift.size() == (B, arch.G ** 2, 2))
            and (z_scale.size() == (B, arch.G ** 2, 2))
            and (z_where.size() == (B, arch.G ** 2, 4))
        )

        return (
            z_pres,
            z_depth,
            z_scale,
            z_shift,
            z_where,
            z_pres_logits,
            z_depth_post,
            z_scale_post,
            z_shift_post,
        )


class ZWhatEnc(nn.Module):
    def __init__(self):
        super(ZWhatEnc, self).__init__()

        self.enc_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 256, 4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
        )

        self.enc_what = nn.Linear(256, arch.z_what_dim * 2)

    def forward(self, x):
        """
        Encode a (32, 32) glimpse into z_what

        :param x: (B, C, H, W)
        :return:
            z_what: (B, D)
            z_what_post: (B, D)
        """
        x = self.enc_cnn(x)

        # (B, D), (B, D)
        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        z_what_std = F.softplus(z_what_std)
        z_what_post = Normal(z_what_mean, z_what_std)
        z_what = z_what_post.rsample()

        return z_what, z_what_post


class GlimpseDec(nn.Module):
    """Decoder z_what into reconstructed objects"""

    def __init__(self):
        super(GlimpseDec, self).__init__()

        # I am using really deep network here. But this is overkill
        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_what_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
        )

        self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)

        self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        """
        Decoder z_what into glimpse

        :param x: (B, D)
        :return:
            o_att: (B, 3, H, W)
            alpha_att: (B, 1, H, W)
        """
        x = self.dec(x.view(x.size(0), -1, 1, 1))

        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))

        return o, alpha


class SpaceBg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.image_enc = ImageEncoderBg()

        # Compute mask hidden states given image features
        self.rnn_mask = nn.LSTMCell(
            arch.z_mask_dim + arch.img_enc_dim_bg, arch.rnn_mask_hidden_dim
        )
        self.rnn_mask_h = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))
        self.rnn_mask_c = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))

        # Dummy z_mask for first step of rnn_mask
        self.z_mask_0 = nn.Parameter(torch.zeros(arch.z_mask_dim))
        # Predict mask latent given h
        self.predict_mask = PredictMask()
        # Compute masks given mask latents
        self.mask_decoder = MaskDecoder()
        # Encode mask and image into component latents
        self.comp_encoder = CompEncoder()
        # Component decoder
        if arch.K > 1:
            self.comp_decoder = CompDecoder()
        else:
            self.comp_decoder = CompDecoderStrong()

        # ==== Prior related ====
        self.rnn_mask_prior = nn.LSTMCell(
            arch.z_mask_dim, arch.rnn_mask_prior_hidden_dim
        )
        # Initial h and c
        self.rnn_mask_h_prior = nn.Parameter(
            torch.zeros(arch.rnn_mask_prior_hidden_dim)
        )
        self.rnn_mask_c_prior = nn.Parameter(
            torch.zeros(arch.rnn_mask_prior_hidden_dim)
        )
        # Compute mask latents
        self.predict_mask_prior = PredictMask()
        # Compute component latents
        self.predict_comp_prior = PredictComp()
        # ==== Prior related ====

        self.bg_sigma = arch.bg_sigma

    def anneal(self, global_step):
        pass

    def forward(self, x, global_step):
        """
        Background inference backward pass

        :param x: shape (B, C, H, W)
        :param global_step: global training step
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization
        """
        B = x.size(0)

        # (B, D)
        x_enc = self.image_enc(x)

        # Mask and component latents over the K slots
        masks = []
        z_masks = []
        # These two are Normal instances
        z_mask_posteriors = []
        z_comp_posteriors = []

        # Initialization: encode x and dummy z_mask_0
        z_mask = self.z_mask_0.expand(B, arch.z_mask_dim)
        h = self.rnn_mask_h.expand(B, arch.rnn_mask_hidden_dim)
        c = self.rnn_mask_c.expand(B, arch.rnn_mask_hidden_dim)

        for i in range(arch.K):
            # Encode x and z_{mask, 1:k}, (b, D)
            rnn_input = torch.cat((z_mask, x_enc), dim=1)
            (h, c) = self.rnn_mask(rnn_input, (h, c))

            # Predict next mask from x and z_{mask, 1:k-1}
            z_mask_loc, z_mask_scale = self.predict_mask(h)
            z_mask_post = Normal(z_mask_loc, z_mask_scale)
            z_mask = z_mask_post.rsample()
            z_masks.append(z_mask)
            z_mask_posteriors.append(z_mask_post)
            # Decode masks
            mask = self.mask_decoder(z_mask)
            masks.append(mask)

        # (B, K, 1, H, W), in range (0, 1)
        masks = torch.stack(masks, dim=1)

        # SBP to ensure they sum to 1
        masks = self.SBP(masks)
        # An alternative is to use softmax
        # masks = F.softmax(masks, dim=1)

        B, K, _, H, W = masks.size()

        # Reshape (B, K, 1, H, W) -> (B*K, 1, H, W)
        masks = masks.view(B * K, 1, H, W)

        # Concatenate images (B*K, 4, H, W)
        comp_vae_input = torch.cat(
            (
                (masks + 1e-5).log(),
                x[:, None].repeat(1, K, 1, 1, 1).view(B * K, 3, H, W),
            ),
            dim=1,
        )

        # Component latents, each (B*K, L)
        z_comp_loc, z_comp_scale = self.comp_encoder(comp_vae_input)
        z_comp_post = Normal(z_comp_loc, z_comp_scale)
        z_comp = z_comp_post.rsample()

        # Record component posteriors here. We will use this for computing KL
        z_comp_loc_reshape = z_comp_loc.view(B, K, -1)
        z_comp_scale_reshape = z_comp_scale.view(B, K, -1)
        for i in range(arch.K):
            z_comp_post_this = Normal(
                z_comp_loc_reshape[:, i], z_comp_scale_reshape[:, i]
            )
            z_comp_posteriors.append(z_comp_post_this)

        # Decode into component images, (B*K, 3, H, W)
        comps = self.comp_decoder(z_comp)

        # Reshape (B*K, ...) -> (B, K, 3, H, W)
        comps = comps.view(B, K, 3, H, W)
        masks = masks.view(B, K, 1, H, W)

        # Now we are ready to compute the background likelihoods
        # (B, K, 3, H, W)
        comp_dist = Normal(comps, torch.full_like(comps, self.bg_sigma))
        log_likelihoods = comp_dist.log_prob(x[:, None].expand_as(comps))

        # (B, K, 3, H, W) -> (B, 3, H, W), mixture likelihood
        log_sum = log_likelihoods + (masks + 1e-5).log()
        bg_likelihood = torch.logsumexp(log_sum, dim=1)

        # Background reconstruction
        bg = (comps * masks).sum(dim=1)

        # Below we compute priors and kls

        # Conditional KLs
        z_mask_total_kl = 0.0
        z_comp_total_kl = 0.0

        # Initial h and c. This is h_1 and c_1 in the paper
        h = self.rnn_mask_h_prior.expand(B, arch.rnn_mask_prior_hidden_dim)
        c = self.rnn_mask_c_prior.expand(B, arch.rnn_mask_prior_hidden_dim)

        for i in range(arch.K):
            # Compute prior distribution over z_masks
            z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(h)
            z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
            # Compute component prior, using posterior samples
            z_comp_loc_prior, z_comp_scale_prior = self.predict_comp_prior(z_masks[i])
            z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
            # Compute KLs as we go.
            z_mask_kl = kl_divergence(z_mask_posteriors[i], z_mask_prior).sum(dim=1)
            z_comp_kl = kl_divergence(z_comp_posteriors[i], z_comp_prior).sum(dim=1)
            # (B,)
            z_mask_total_kl += z_mask_kl
            z_comp_total_kl += z_comp_kl

            # Compute next state. Note we condition we posterior samples.
            # Again, this is conditional prior.
            (h, c) = self.rnn_mask_prior(z_masks[i], (h, c))

        # For visualization
        kl_bg = z_mask_total_kl + z_comp_total_kl
        log = {
            # (B, K, 3, H, W)
            "comps": comps,
            # (B, 1, 3, H, W)
            "masks": masks,
            # (B, 3, H, W)
            "bg": bg,
            "kl_bg": kl_bg,
        }

        return bg_likelihood, bg, kl_bg, log

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, K, _, H, W = masks.size()

        # (B, 1, H, W)
        remained = torch.ones_like(masks[:, 0])
        # remained = torch.ones_like(masks[:, 0]) - fg_mask
        new_masks = []
        for k in range(K):
            if k < K - 1:
                mask = masks[:, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)

        new_masks = torch.stack(new_masks, dim=1)

        return new_masks


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageEncoderBg(nn.Module):
    """Background image encoder"""

    def __init__(self):
        embed_size = arch.img_shape[0] // 16
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 16x downsampled: (64, H/16, W/16)
            Flatten(),
            nn.Linear(64 * embed_size ** 2, arch.img_enc_dim_bg),
            nn.ELU(),
        )

    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, D)
        """
        return self.enc(x)


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(arch.rnn_mask_hidden_dim, arch.z_mask_dim * 2)

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference

        :param h: hidden state from rnn_mask
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)

        """
        x = self.fc(h)
        z_mask_loc = x[:, : arch.z_mask_dim]
        z_mask_scale = F.softplus(x[:, arch.z_mask_dim :]) + 1e-4

        return z_mask_loc, z_mask_scale


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""

    def __init__(self):
        super(MaskDecoder, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_mask_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def forward(self, z_mask):
        """
        Decode z_mask into mask

        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        B = z_mask.size(0)
        # 1d -> 3d, (B, D, 1, 1)
        z_mask = z_mask.view(B, -1, 1, 1)
        mask = torch.sigmoid(self.dec(z_mask))
        return mask


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """

    def __init__(self):
        nn.Module.__init__(self)

        embed_size = arch.img_shape[0] // 16
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Flatten(),
            # 16x downsampled: (64, 4, 4)
            nn.Linear(64 * embed_size ** 2, arch.z_comp_dim * 2),
        )

    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated

        :param x: (B, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.enc(x)
        z_comp_loc = x[:, : arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim :]) + 1e-4

        return z_comp_loc, z_comp_scale


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].expand(B, 2, height, width)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)
        self.decoder = nn.Sequential(
            nn.Conv2d(arch.z_comp_dim + 2, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 16x downsampled: (32, 4, 4)
            nn.Conv2d(32, 3, 1, 1),
        )

    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = arch.img_shape
        # (B, L) -> (B, L+2, H, W)
        z_comp = self.spatial_broadcast(z_comp, h + 8, w + 8)
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):
    def __init__(self):
        super(CompDecoderStrong, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_comp_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, x):
        """

        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.sigmoid(self.dec(x))
        return comp


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(arch.z_mask_dim, arch.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(arch.predict_comp_hidden_dim, arch.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(arch.predict_comp_hidden_dim, arch.z_comp_dim * 2),
        )

    def forward(self, h):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, : arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim :]) + 1e-4

        return z_comp_loc, z_comp_scale


class Space(nn.Module):
    shortname = "space"

    def __init__(self):
        nn.Module.__init__(self)

        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()

    def forward(self, x, global_step=None):
        """
        Inference.

        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B,)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)

        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(
            x, global_step
        )

        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)

        # Compute final mixture likelhood
        # (B, 3, H, W)
        fg_likelihood = fg_likelihood + (alpha_map + 1e-5).log()
        bg_likelihood = bg_likelihood + (1 - alpha_map + 1e-5).log()
        # (B, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        y = alpha_map * fg + (1.0 - alpha_map) * bg

        # Elbo
        kl = kl_bg + kl_fg
        elbo = log_like - kl

        # Mean over batch
        loss = (-elbo + loss_boundary).mean()

        log = {
            "imgs": x,
            "y": y,
            # (B,)
            "mse": ((y - x) ** 2).flatten(start_dim=1).sum(dim=1),
            "log_like": log_like,
        }
        log.update(log_fg)
        log.update(log_bg)

        fg_box = bbox_in_one(log["fg"], log["z_pres"], log["z_scale"], log["z_shift"])
        log.update({"fg_box": fg_box})
        torch.any(fg_box > 0.0, dim=1)
        if torch.any(torch.isnan(loss) or torch.isinf(loss)):
            import ipdb

            ipdb.set_trace()
        ret = {
            "loss": loss,
            "rec_loss": log_like.mean(),
            "boundary_loss": loss_boundary.mean(),
            "elbo": elbo.mean(),
            "kl": kl.mean(),
            "kl_fg": kl_fg.mean(),
            "kl_bg": kl_bg.mean(),
            "kl_what": log["kl_z_what"].mean(),
            "kl_where": log["kl_z_where"].mean(),
            "kl_depth": log["kl_z_depth"].mean(),
            "kl_pres": log["kl_z_pres"].mean(),
            "canvas": y,
            "canvas_with_bbox": fg_box,
            "background": bg,
            "steps": {
                "patch": log["patches"],
                "mask": log["patch_masks"],
                "z_pres": log["z_pres"],
            },
            "layers": {"patch": log["comps"], "mask": log["masks"]},
            "counts": torch.round(log["z_pres"]).flatten(1).sum(-1),
        }

        return ret


gbox = torch.zeros(3, 21, 21)
gbox[1, :2, :] = 1
gbox[1, -2:, :] = 1
gbox[1, :, :2] = 1
gbox[1, :, -2:] = 1
gbox = gbox.view(1, 3, 21, 21)


def bbox_in_one(x, z_pres, z_where_scale, z_where_shift, gbox=gbox):
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    z_pres = z_pres.reshape(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(-1, 2)
    z_shift = z_where_shift.reshape(-1, 2)
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    bbox = spatial_transform(
        z_pres * gbox.to(z_pres.device),  # + (1 - z_pres) * rbox,
        torch.cat((z_scale, z_shift), dim=1),
        torch.Size([B * N, 3, *img_shape]),
        inverse=True,
    )
    bbox = (bbox.reshape(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0) + x).clamp(
        0.0, 1.0
    )
    return bbox
