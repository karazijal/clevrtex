"""
Genesis-V2
https://arxiv.org/pdf/2104.09958

Based on original implementation from
https://github.com/martinengelcke/genesis/blob/genesis-v2/models/genesisv2_config.py


# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np


class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True),
        )


class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        c = torch.linspace(-1, 1, dim)
        self.register_buffer("coords", torch.stack(torch.meshgrid(c, c))[None])

    def forward(self, x):
        b_sz = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return torch.cat([x, self.coords.expand(b_sz, -1, -1, -1)], dim=1)


class UNet(nn.Module):
    def __init__(
        self, num_blocks, img_size=64, filter_start=32, in_chnls=4, out_chnls=1
    ):
        super(UNet, self).__init__()
        c = filter_start
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2 * c, 2 * c]
            enc_out = [c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2 * c, 2 * c]
            enc_out = [c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2 * c, 2 * c]
            enc_out = [c, c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c, c]
        self.down = []
        self.up = []
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(ConvGNReLU(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(ConvGNReLU(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2 ** (num_blocks - 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * c * self.featuremap_size ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * c * self.featuremap_size ** 2),
            nn.ReLU(),
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv2d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()
        self.out_chnls = out_chnls

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode="nearest")
            x_down.append(act)
        # FC
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1, self.featuremap_size, self.featuremap_size)
        # Up
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode="nearest")
        return self.final_conv(x_up), None


@torch.no_grad()
def check_log_masks(log_m_k):
    # Note: this seems to be checking for under(over)flow when moving log->exp space
    # Adjusted this to run on GPU most of the time until the errors need to be printed
    flat = torch.stack(log_m_k, dim=4).exp().sum(dim=4).view(-1)
    diff = flat - torch.ones_like(flat)
    max_diff, idx = diff.max(dim=0)
    if torch.anu(max_diff > 1e-3) or torch.any(torch.isnan(flat)):
        print(f"Max diff: {max_diff.cpu().item()}")
        masks_k = log_m_k.view(log_m_k.shape[0], -1)[:, idx].exp().cpu().squeeze()
        for i, v in enumerate(masks_k):
            print(f"Mask value at k={i}: {v}")
        # TODO: change this drop into an interactive debugger if interactive env.
        raise ValueError("Masks do not sum to 1.0. Not close enough.")


class SemiConv(nn.Conv2d):
    def __init__(self, in_c, out_ch, size):
        super(SemiConv, self).__init__(in_c, out_ch, 1)
        self.gate = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.register_buffer(
            "uv",
            torch.cat(
                [
                    torch.zeros(out_ch - 2, size, size),
                    torch.stack(torch.meshgrid(*([torch.linspace(-1, 1, size)] * 2))),
                ]
            )[None],
            persistent=False,
        )

    def forward(self, input):
        x = self.gate * super(SemiConv, self).forward(input)
        return x + self.uv, x[:, -2:]


def clamp_st(x, lower, upper):
    # From: http://docs.pyro.ai/en/0.3.3/_modules/pyro/distributions/iaf.html
    return x + (x.clamp(lower, upper) - x).detach()


def euclidian_norm(x):
    # Clamp before taking sqrt for numerical stability
    return clamp_st((x ** 2).sum(1), 1e-10, 1e10).sqrt()


def euclidian_distance(ea, eb):
    # Unflatten if needed if one is an image and the other a vector
    if ea.dim() == 4 and eb.dim() == 2:
        eb = eb.unsqueeze(-1).unsqueeze(-1)
    if eb.dim() == 4 and ea.dim() == 2:
        ea = ea.unsqueeze(-1).unsqueeze(-1)
    return euclidian_norm(ea - eb)


def squared_distance(ea, eb):
    if ea.dim() == 4 and eb.dim() == 2:
        eb = eb.unsqueeze(-1).unsqueeze(-1)
    if eb.dim() == 4 and ea.dim() == 2:
        ea = ea.unsqueeze(-1).unsqueeze(-1)
    return ((ea - eb) ** 2).sum(1)


class InstanceColouringSBP(nn.Module):
    def __init__(
        self,
        img_size,
        kernel="gaussian",
        colour_dim=8,
        K_steps=None,
        feat_dim=None,
        semiconv=True,
    ):
        super(InstanceColouringSBP, self).__init__()
        # Config
        self.img_size = img_size
        self.kernel = kernel
        self.colour_dim = colour_dim
        # Initialise kernel sigma
        if self.kernel == "laplacian":
            sigma_init = 1.0 / (np.sqrt(K_steps) * np.log(2))
        elif self.kernel == "gaussian":
            sigma_init = 1.0 / (K_steps * np.log(2))
        elif self.kernel == "epanechnikov":
            sigma_init = 2.0 / K_steps
        else:
            return ValueError("No valid kernel.")
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        # Colour head
        if semiconv:
            self.colour_head = SemiConv(feat_dim, self.colour_dim, img_size)
        else:
            self.colour_head = nn.Conv2d(feat_dim, self.colour_dim, 1)

    def forward(
        self, features, steps_to_run, debug=False, dynamic_K=False, *args, **kwargs
    ):
        batch_size = features.size(0)
        if dynamic_K:
            assert batch_size == 1
        # Get colours
        colour_out = self.colour_head(features)
        if isinstance(colour_out, tuple):
            colour, delta = colour_out
        else:
            colour, delta = colour_out, None
        # Sample from uniform to select random pixels as seeds
        # rand_pixel = torch.empty(batch_size, 1, *colour.shape[2:])
        # rand_pixel = rand_pixel.uniform_()
        rand_pixel = torch.rand(
            batch_size, 1, *colour.shape[2:], device=features.device
        )
        # Run SBP
        seed_list = []
        log_m_k = []
        log_s_k = [
            torch.zeros(
                batch_size, 1, self.img_size, self.img_size, device=features.device
            )
        ]
        for step in range(steps_to_run):
            # Determine seed
            scope = F.interpolate(
                log_s_k[step].exp(),
                size=colour.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pixel_probs = rand_pixel * scope
            rand_max = pixel_probs.flatten(2).argmax(2).flatten()
            # --TODO(martin): parallelise this--
            seed = features.new_empty((batch_size, self.colour_dim))
            for bidx in range(batch_size):
                seed[bidx, :] = colour.flatten(2)[bidx, :, rand_max[bidx]]
            # seed = colour.flatten(2).gather(-1, rand_max.view(-1, 1, 1).expand(-1, rand_max.shape[1], -1)).squeeze(-1)
            seed_list.append(seed)
            # Compute masks
            # Note the distance here is in channel-wise
            if self.kernel == "laplacian":
                distance = euclidian_distance(colour, seed)  # [B, H, W]
                alpha = torch.exp(-distance / self.log_sigma.exp())
            elif self.kernel == "gaussian":
                distance = squared_distance(colour, seed)  # [B, H, W]
                alpha = torch.exp(-distance / self.log_sigma.exp())
            elif self.kernel == "epanechnikov":
                distance = squared_distance(colour, seed)  # [B, H, W]
                alpha = (1 - distance / self.log_sigma.exp()).relu()
            else:
                raise ValueError("No valid kernel.")
            alpha = alpha.unsqueeze(1)
            # Sanity checks
            if debug:
                assert alpha.max() <= 1, alpha.max()
                assert alpha.min() >= 0, alpha.min()
            # Clamp mask values to [0.01, 0.99] for numerical stability
            # TODO(martin): clamp less aggressively?
            alpha = clamp_st(alpha, 0.01, 0.99)
            # SBP update
            log_a = torch.log(alpha)
            log_neg_a = torch.log(1 - alpha)
            log_m = log_s_k[step] + log_a

            if dynamic_K and log_m.exp().sum() < 20:
                break
            log_m_k.append(log_m)
            log_s_k.append(log_s_k[step] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Accumulate stats
        stats = {"colour": colour, "delta": delta, "seeds": seed_list}
        return log_m_k, log_s_k, stats


def genesis_x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
    # 1.) Sum over steps for per pixel & channel (ppc) losses
    p_xr_stack = dist.Normal(torch.stack(x_r_k, dim=4), std)
    log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
    log_m_stack = torch.stack(log_m_k, dim=4)
    log_mx = log_m_stack + log_xr_stack
    err_ppc = -log_mx.logsumexp(dim=4)
    # 2.) Sum accross channels and spatial dimensions
    if pixel_wise:
        return err_ppc
    else:
        return err_ppc.sum(dim=(1, 2, 3))


def genesis_mask_latent_loss(
    q_zm_0_k,
    zm_0_k,
    zm_k_k=None,
    ldj_k=None,
    prior_lstm=None,
    prior_linear=None,
    debug=False,
):
    num_steps = len(zm_0_k)
    batch_size = zm_0_k[0].size(0)
    latent_dim = zm_0_k[0].size(1)
    if zm_k_k is None:
        zm_k_k = zm_0_k

    # -- Determine prior --
    if prior_lstm is not None and prior_linear is not None:
        # zm_seq shape: (att_steps-2, batch_size, ldim)
        # Do not need the last element in z_k
        zm_seq = torch.cat([zm.view(1, batch_size, -1) for zm in zm_k_k[:-1]], dim=0)
        # lstm_out shape: (att_steps-2, batch_size, state_size)
        # Note: recurrent state is handled internally by LSTM
        lstm_out, _ = prior_lstm(zm_seq)
        # linear_out shape: (att_steps-2, batch_size, 2*ldim)
        linear_out = prior_linear(lstm_out)
        linear_out = torch.chunk(linear_out, 2, dim=2)
        mu_raw = torch.tanh(linear_out[0])
        # Note: ditton about prior_linear
        sigma_raw = torch.sigmoid(linear_out[1] + 4.0) + 1e-4
        # Split into K steps, shape: (att_steps-2)*[1, batch_size, ldim]
        mu_k = torch.split(mu_raw, 1, dim=0)
        sigma_k = torch.split(sigma_raw, 1, dim=0)
        # Use standard Normal as prior for first step
        p_zm_k = [dist.Normal(0, 1)]
        # Autoregressive prior for later steps
        for mean, std in zip(mu_k, sigma_k):
            # Remember to remove unit dimension at dim=0
            p_zm_k += [
                dist.Normal(
                    mean.view(batch_size, latent_dim), std.view(batch_size, latent_dim)
                )
            ]
        # Sanity checks
        if debug:
            assert zm_seq.size(0) == num_steps - 1
    else:
        p_zm_k = num_steps * [dist.Normal(0, 1)]

    # -- Compute KL using Monte Carlo samples for every step k --
    kl_m_k = []
    for step, p_zm in enumerate(p_zm_k):
        log_q = q_zm_0_k[step].log_prob(zm_0_k[step]).sum(dim=1)
        log_p = p_zm.log_prob(zm_k_k[step]).sum(dim=1)
        kld = log_q - log_p
        if ldj_k is not None:
            ldj = ldj_k[step].sum(dim=1)
            kld = kld - ldj
        kl_m_k.append(kld)

    # -- Sanity check --
    if debug:
        assert len(p_zm_k) == num_steps
        assert len(kl_m_k) == num_steps

    return kl_m_k, p_zm_k


def monet_get_mask_recon_stack(m_r_logits_k, prior_mode, log):
    if prior_mode == "softmax":
        if log:
            return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
    elif prior_mode == "scope":
        log_m_r_k = []
        log_s = torch.zeros_like(m_r_logits_k[0])
        for step, logits in enumerate(m_r_logits_k):
            if step == len(m_r_logits_k) - 1:
                log_m_r_k.append(log_s)
            else:
                log_a = F.logsigmoid(logits)
                log_neg_a = F.logsigmoid(-logits)
                log_m_r_k.append(log_s + log_a)
                log_s = log_s + log_neg_a
        log_m_r_stack = torch.stack(log_m_r_k, dim=4)
        return log_m_r_stack if log else log_m_r_stack.exp()
    else:
        raise ValueError("No valid prior mode.")


def monet_kl_m_loss(log_m_k, log_m_r_k, debug=False):
    if debug:
        assert len(log_m_k) == len(log_m_r_k)
    batch_size = log_m_k[0].size(0)
    m_stack = torch.stack(log_m_k, dim=4).exp()
    m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
    # Lower bound to 1e-5 to avoid infinities
    m_stack = torch.max(m_stack, torch.tensor(1e-5))
    m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5))
    q_m = dist.Categorical(m_stack.view(-1, len(log_m_k)))
    p_m = dist.Categorical(m_r_stack.view(-1, len(log_m_k)))
    kl_m_ppc = dist.kl_divergence(q_m, p_m).view(batch_size, -1)
    return kl_m_ppc.sum(dim=1)


class GenesisV2(nn.Module):
    shortname = "g2"

    def __init__(
        self,
        feat_dim=64,
        kernel="gaussian",
        semiconv=True,
        dynamic_K=False,
        klm_loss=False,
        detach_mr_in_klm=True,
        g_goal=0.5655,
        g_lr=1e-5,
        g_alpha=0.99,
        g_init=1.0,
        g_min=1e-10,
        g_speedup=10.0,
        K_steps=11,
        img_size=128,  # Clevr
        autoreg_prior=True,
        pixel_bound=True,
        pixel_std=0.7,
        debug=False,
    ):
        super(GenesisV2, self).__init__()
        # Configuration
        self.K_steps = K_steps
        self.pixel_bound = pixel_bound
        self.feat_dim = feat_dim
        self.klm_loss = klm_loss
        self.detach_mr_in_klm = detach_mr_in_klm
        self.dynamic_K = dynamic_K
        self.debug = debug
        # Encoder
        self.encoder = UNet(
            num_blocks=int(np.log2(img_size) - 1),
            img_size=img_size,
            filter_start=min(feat_dim, 64),
            in_chnls=3,
            out_chnls=-1,
        )
        self.att_process = InstanceColouringSBP(
            img_size=img_size,
            kernel=kernel,
            colour_dim=8,
            K_steps=self.K_steps,
            feat_dim=feat_dim,
            semiconv=semiconv,
        )
        self.seg_head = ConvGNReLU(feat_dim, feat_dim, 3, 1, 1)
        self.feat_head = nn.Sequential(
            ConvGNReLU(feat_dim, feat_dim, 3, 1, 1),
            nn.Conv2d(feat_dim, 2 * feat_dim, 1),
        )
        self.z_head = nn.Sequential(
            nn.LayerNorm(2 * feat_dim),
            nn.Linear(2 * feat_dim, 2 * feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * feat_dim, 2 * feat_dim),
        )
        # Decoder
        c = feat_dim
        self.decoder_module = nn.Sequential(
            BroadcastLayer(img_size // 16),
            nn.ConvTranspose2d(feat_dim + 2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1),
        )
        # --- Prior ---
        self.autoreg_prior = autoreg_prior
        self.prior_lstm, self.prior_linear = None, None
        if self.autoreg_prior and self.K_steps > 1:
            self.prior_lstm = nn.LSTM(feat_dim, 4 * feat_dim)
            self.prior_linear = nn.Linear(4 * feat_dim, 2 * feat_dim)
        # --- Output pixel distribution ---
        # assert pixel_std1 == cfg.pixel_std2
        self.std = pixel_std
        self.geco = GECO(
            g_goal * 3 * img_size ** 2,
            g_lr * (64 ** 2 / img_size ** 2),
            g_alpha,
            g_init,
            g_min,
            g_speedup,
        )

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # --- Extract features ---
        enc_feat, _ = self.encoder(x)
        enc_feat = F.relu(enc_feat)

        # --- Predict attention masks ---
        if self.dynamic_K:
            if batch_size > 1:
                # Iterate over individual elements in batch
                log_m_k = [[] for _ in range(self.K_steps)]
                att_stats, log_s_k = None, None
                for f in torch.split(enc_feat, 1, dim=0):
                    log_m_k_b, _, _ = self.att_process(
                        self.seg_head(f), self.K_steps - 1, dynamic_K=True
                    )
                    for step in range(self.K_steps):
                        if step < len(log_m_k_b):
                            log_m_k[step].append(log_m_k_b[step])
                        else:
                            log_m_k[step].append(-1e10 * torch.ones([1, 1, H, W]))
                for step in range(self.K_steps):
                    log_m_k[step] = torch.cat(log_m_k[step], dim=0)
                if self.debug:
                    assert len(log_m_k) == self.K_steps
            else:
                log_m_k, log_s_k, att_stats = self.att_process(
                    self.seg_head(enc_feat), self.K_steps - 1, dynamic_K=True
                )
        else:
            log_m_k, log_s_k, att_stats = self.att_process(
                self.seg_head(enc_feat), self.K_steps - 1, dynamic_K=False
            )
            if self.debug:
                assert len(log_m_k) == self.K_steps

        # -- Object features, latents, and KL
        comp_stats = dict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        for log_m in log_m_k:
            mask = log_m.exp()
            # Masked sum
            obj_feat = mask * self.feat_head(enc_feat)
            obj_feat = obj_feat.sum((2, 3))
            # Normalise
            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-5)
            # Posterior
            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            # Note: Not sure why sigma needs to biased by 0.5 here; leaving though
            sigma = F.softplus(sigma_ps + 0.5) + 1e-8
            q_z = dist.Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats["mu_k"].append(mu)
            comp_stats["sigma_k"].append(sigma)
            comp_stats["z_k"].append(z)
            comp_stats["q_z_k"].append(q_z)

        # --- Decode latents ---
        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats["z_k"])

        # --- Loss terms ---
        losses = {}
        # -- Reconstruction loss
        losses["err"] = genesis_x_loss(x, log_m_r_k, x_r_k, self.std)
        mx_r_k = [x * logm.exp() for x, logm in zip(x_r_k, log_m_r_k)]
        # -- Optional: Attention mask loss
        if self.klm_loss:
            if self.detach_mr_in_klm:
                log_m_r_k = [m.detach() for m in log_m_r_k]
            losses["kl_m"] = monet_kl_m_loss(
                log_m_k=log_m_k, log_m_r_k=log_m_r_k, debug=self.debug
            )
        # -- Component KL
        losses["kl_l_k"], p_z_k = genesis_mask_latent_loss(
            comp_stats["q_z_k"],
            comp_stats["z_k"],
            prior_lstm=self.prior_lstm,
            prior_linear=self.prior_linear,
            debug=self.debug,
        )

        # Track quantities of interest
        stats = dict(
            recon=recon,
            log_m_k=log_m_k,
            log_s_k=log_s_k,
            x_r_k=x_r_k,
            log_m_r_k=log_m_r_k,
            mx_r_k=mx_r_k,
            instance_seg=torch.argmax(torch.cat(log_m_k, dim=1), dim=1),
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k, dim=1), dim=1),
        )

        # Sanity checks
        if self.debug:
            if not self.dynamic_K:
                assert len(log_m_k) == self.K_steps
                assert len(log_m_r_k) == self.K_steps
            check_log_masks(log_m_k)
            check_log_masks(log_m_r_k)

        recon_loss = losses["err"].mean()
        kl_m, kl_l = torch.tensor(0), torch.tensor(0)
        # -- KL stage 1
        if "kl_m" in losses:
            kl_m = losses["kl_m"].mean(0)
        elif "kl_m_k" in losses:
            kl_m = torch.stack(losses["kl_m_k"], dim=1).mean(dim=0).sum()
        # -- KL stage 2
        if "kl_l" in losses:
            kl_l = losses["kl_l"].mean(0)
        elif "kl_l_k" in losses:
            kl_l = torch.stack(losses["kl_l_k"], dim=1).mean(dim=0).sum()
        kl = (kl_l + kl_m).mean(0)

        elbo = recon_loss + kl
        loss = self.geco.loss(recon_loss, kl)

        ret = {
            "canvas": recon,
            "loss": loss,
            "elbo": elbo,
            "rec_loss": recon_loss,
            "kl": kl,
            "beta": self.geco.beta,
            "layers": {
                "mask": torch.stack(log_m_r_k, dim=1).exp(),
                "patch": torch.stack(x_r_k, dim=1),
                "other_mask": torch.stack(log_m_k, dim=1).exp(),
            },
        }

        return ret

    def decode_latents(self, z_k):
        # --- Reconstruct components and image ---
        x_r_k, m_r_logits_k = [], []
        for z in z_k:
            dec = self.decoder_module(z)
            x_r_k.append(dec[:, :3, :, :])
            m_r_logits_k.append(dec[:, 3:, :, :])
        # Optional: Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        # --- Reconstruct masks ---
        log_m_r_stack = monet_get_mask_recon_stack(m_r_logits_k, "softmax", log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps

        # Sample latents
        if self.autoreg_prior:
            z_k = [dist.Normal(0, 1).sample([batch_size, self.feat_dim])]
            state = None
            for k in range(1, K_steps):
                # TODO(martin): reuse code from forward method?
                lstm_out, state = self.prior_lstm(
                    z_k[-1].view(1, batch_size, -1), state
                )
                linear_out = self.prior_linear(lstm_out)
                linear_out = torch.chunk(linear_out, 2, dim=2)
                linear_out = [item.squeeze(0) for item in linear_out]
                mu = torch.tanh(linear_out[0])
                # Note: the 4.0 bias seems to be for 0-initialised self.prior_linear.bias.
                # This gives starting sigma as 1.0001.
                # TODO: replace this with something else, like a direct init of bias.
                # TODO: Sigmoid saturates outside of (-88, 16) (0. gradients, 0. or 1. output).
                # TODO: Having a positive target +4.0 seems to somewhat reduce that.
                sigma = torch.sigmoid(linear_out[1] + 4.0) + 1e-4
                p_z = dist.Normal(
                    mu.view([batch_size, self.feat_dim]),
                    sigma.view([batch_size, self.feat_dim]),
                )
                z_k.append(p_z.sample())
        else:
            p_z = dist.Normal(0, 1)
            z_k = [p_z.sample([batch_size, self.feat_dim]) for _ in range(K_steps)]

        # Decode latents
        recon, x_r_k, log_m_r_k = self.decode_latents(z_k)

        stats = dict(
            x_k=x_r_k,
            log_m_k=log_m_r_k,
            mx_k=[x * m.exp() for x, m in zip(x_r_k, log_m_r_k)],
        )
        return recon, stats


class GECO(nn.Module):
    def __init__(
        self, goal, step_size, alpha=0.99, beta_init=1.0, beta_min=1e-10, speedup=None
    ):
        super(GECO, self).__init__()
        self.err_ema = None
        # self.goal = goal
        # self.step_size = step_size
        # self.alpha = alpha
        # self.beta = torch.tensor(beta_init)
        # self.beta_min = torch.tensor(beta_min)
        # self.beta_max = torch.tensor(1e10)
        # self.speedup = speedup
        self.register_buffer("goal", torch.tensor(goal))
        self.register_buffer("step_size", torch.tensor(step_size))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta_init))
        self.register_buffer("beta_min", torch.tensor(beta_min))
        self.register_buffer("beta_max", torch.tensor(1e10))
        if speedup is not None:
            self.register_buffer("speedup", torch.tensor(speedup))

    def to_cuda(self):
        self.beta = self.beta.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema
            constraint = self.goal - self.err_ema
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
