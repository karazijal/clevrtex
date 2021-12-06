"""
Code adjusted from https://github.com/JindongJiang/GNM

"Generative Neurosymbolic Machines"
Jindong Jiang & Sungjin Ahn
NeurIPS 2020
https://arxiv.org/abs/2010.12152
"""
from typing import List, Tuple, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


def linear_schedule_tensor(step, start_step, end_step, start_value, end_value, device):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step)).to(device)
    elif step >= end_step:
        x = torch.tensor(end_value).to(device)
    else:
        x = torch.tensor(start_value).to(device)

    return x


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
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-15)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-15)

    # set translation
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-15)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-15)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)


def kl_divergence_bern_bern(q_pres_probs, p_pres_prob, eps=1e-15):
    """
    Compute kl divergence
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    # z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = q_pres_probs * (torch.log(q_pres_probs + eps) - torch.log(p_pres_prob + eps)) + \
         (1 - q_pres_probs) * (torch.log(1 - q_pres_probs + eps) - torch.log(1 - p_pres_prob + eps))

    return kl


class StackConvNorm(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 groupings: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU):
        super(StackConvNorm, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            if s == 0:
                layers.append(nn.Conv2d(dim_prev, f, k, 1, 0))
            else:
                layers.append(nn.Conv2d(dim_prev, f, k, s, (k - 1) // 2))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            layers.append(nn.GroupNorm(groupings[i], f))
            # layers.append(nn.BatchNorm2d(f))
            dim_prev = f

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x


class StackSubPixelNorm(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 kernel_sizes: List[int],
                 upscale: List[int],
                 groupings: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU):
        super(StackSubPixelNorm, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, (f, k, u) in enumerate(zip(filters, kernel_sizes, upscale)):
            if u == 1:
                layers.append(nn.Conv2d(dim_prev, f, k, 1, (k - 1) // 2))
            else:
                layers.append(nn.Conv2d(dim_prev, f * u ** 2, k, 1, (k - 1) // 2))
                layers.append(nn.PixelShuffle(u))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            layers.append(nn.GroupNorm(groupings[i], f))
            dim_prev = f

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x


class StackMLP(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU,
                 phase_layer_norm: bool = True):
        super(StackMLP, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, f in enumerate(filters):
            layers.append(nn.Linear(dim_prev, f))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            if phase_layer_norm:
                layers.append(nn.LayerNorm(f))
            dim_prev = f

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_cell=4):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=self.input_dim + hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

        self.register_parameter('h_0', torch.nn.Parameter(torch.zeros(1, self.hidden_dim, num_cell, num_cell),
                                                          requires_grad=True))
        self.register_parameter('c_0', torch.nn.Parameter(torch.zeros(1, self.hidden_dim, num_cell, num_cell),
                                                          requires_grad=True))

    def forward(self, x, h_c):
        h_cur, c_cur = h_c

        conv_inp = torch.cat([x, h_cur], dim=1)

        i, f, o, c = self.conv(conv_inp).split(self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        c_next = f * c_cur + i * c
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return self.h_0.expand(batch_size, -1, -1, -1), \
               self.c_0.expand(batch_size, -1, -1, -1)


class LocalLatentDecoder(nn.Module):
    def __init__(self, args: Any):
        super(LocalLatentDecoder, self).__init__()
        self.args = args

        pwdw_net_inp_dim = self.args.arch.img_enc_dim

        self.pwdw_net = StackConvNorm(
            pwdw_net_inp_dim,
            self.args.arch.pwdw.pwdw_filters,
            self.args.arch.pwdw.pwdw_kernel_sizes,
            self.args.arch.pwdw.pwdw_strides,
            self.args.arch.pwdw.pwdw_groups,
            norm_act_final=True
        )

        self.q_depth_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_depth_dim * 2, 1)
        self.q_where_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_where_dim * 2, 1)
        self.q_what_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_what_dim * 2, 1)
        self.q_pres_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_pres_dim, 1)

        torch.nn.init.uniform_(self.q_where_net.weight.data, -0.01, 0.01)
        # scale
        torch.nn.init.constant_(self.q_where_net.bias.data[0], -1.)
        # ratio, x, y, std
        torch.nn.init.constant_(self.q_where_net.bias.data[1:], 0)

    def forward(self, img_enc: torch.Tensor, ss_p_z: List = None) -> List:
        """
        :param img_enc: (bs, dim, 4, 4)
        :param global_dec: (bs, dim, 4, 4)
        :return:
        """

        if ss_p_z is not None:
            p_pres_logits, p_where_mean, p_where_std, p_depth_mean, \
            p_depth_std, p_what_mean, p_what_std = ss_p_z

        pwdw_inp = img_enc

        pwdw_ss = self.pwdw_net(pwdw_inp)

        q_pres_logits = self.q_pres_net(pwdw_ss).tanh() * self.args.const.pres_logit_scale

        # q_where_mean, q_where_std: (bs, dim, num_cell, num_cell)
        q_where_mean, q_where_std = \
            self.q_where_net(pwdw_ss).chunk(2, 1)
        q_where_std = F.softplus(q_where_std)

        # q_depth_mean, q_depth_std: (bs, dim, num_cell, num_cell)
        q_depth_mean, q_depth_std = \
            self.q_depth_net(pwdw_ss).chunk(2, 1)
        q_depth_std = F.softplus(q_depth_std)

        q_what_mean, q_what_std = \
            self.q_what_net(pwdw_ss).chunk(2, 1)
        q_what_std = F.softplus(q_what_std)

        ss = [
            q_pres_logits, q_where_mean, q_where_std,
            q_depth_mean, q_depth_std, q_what_mean, q_what_std
        ]

        return ss


class LocalLatentGenerator(nn.Module):

    def __init__(self, args: Any):
        super(LocalLatentGenerator, self).__init__()
        self.args = args

        self.pwdw_net = StackConvNorm(
            self.args.arch.img_enc_dim,
            self.args.arch.pwdw.pwdw_filters,
            self.args.arch.pwdw.pwdw_kernel_sizes,
            self.args.arch.pwdw.pwdw_strides,
            self.args.arch.pwdw.pwdw_groups,
            norm_act_final=True
        )

        self.p_depth_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_depth_dim * 2, 1)
        self.p_where_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_where_dim * 2, 1)
        self.p_what_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_what_dim * 2, 1)
        self.p_pres_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_pres_dim, 1)

        torch.nn.init.uniform_(self.p_where_net.weight.data, -0.01, 0.01)
        # scale
        torch.nn.init.constant_(self.p_where_net.bias.data[0], -1.)
        # ratio, x, y, std
        torch.nn.init.constant_(self.p_where_net.bias.data[1:], 0)

    def forward(self, global_dec: torch.Tensor) -> List:
        """
        :param x: sample of img_enc (bs, dim, 4, 4)
        :return:
        """

        pwdw_ss = self.pwdw_net(global_dec)

        p_pres_logits = self.p_pres_net(pwdw_ss).tanh() * self.args.const.pres_logit_scale

        # p_where_mean, p_where_std: (bs, dim, num_cell, num_cell)
        p_where_mean, p_where_std = \
            self.p_where_net(pwdw_ss).chunk(2, 1)
        p_where_std = F.softplus(p_where_std)

        # p_depth_mean, p_depth_std: (bs, dim, num_cell, num_cell)
        p_depth_mean, p_depth_std = \
            self.p_depth_net(pwdw_ss).chunk(2, 1)
        p_depth_std = F.softplus(p_depth_std)

        p_what_mean, p_what_std = \
            self.p_what_net(pwdw_ss).chunk(2, 1)
        p_what_std = F.softplus(p_what_std)

        ss = [
            p_pres_logits, p_where_mean, p_where_std,
            p_depth_mean, p_depth_std, p_what_mean, p_what_std
        ]

        return ss


class LocalSampler(nn.Module):

    def __init__(self, args: Any):
        super(LocalSampler, self).__init__()
        self.args = args

        self.z_what_decoder_net = StackSubPixelNorm(
            self.args.z.z_what_dim,
            self.args.arch.conv.p_what_decoder_filters,
            self.args.arch.conv.p_what_decoder_kernel_sizes,
            self.args.arch.conv.p_what_decoder_upscales,
            self.args.arch.conv.p_what_decoder_groups,
            norm_act_final=False
        )

        self.register_buffer('offset', torch.stack(
            torch.meshgrid(torch.arange(args.arch.num_cell).float(),
                           torch.arange(args.arch.num_cell).float())[::-1], dim=0
        ).view(1, 2, args.arch.num_cell, args.arch.num_cell))

    def forward(self, ss: List, phase_use_mode: bool = False) -> Tuple:

        p_pres_logits, p_where_mean, p_where_std, p_depth_mean, \
        p_depth_std, p_what_mean, p_what_std = ss

        if phase_use_mode:
            z_pres = (p_pres_logits > 0).float()
        else:
            z_pres = dist.RelaxedBernoulli(logits=p_pres_logits, temperature=self.args.train.tau_pres).rsample()

        # z_where_scale, z_where_shift: (bs, dim, num_cell, num_cell)
        if phase_use_mode:
            z_where_scale, z_where_shift = p_where_mean.chunk(2, 1)
        else:
            z_where_scale, z_where_shift = \
                dist.Normal(p_where_mean, p_where_std).rsample().chunk(2, 1)

        # z_where_origin: (bs, dim, num_cell, num_cell)
        z_where_origin = \
            torch.cat([z_where_scale.detach(), z_where_shift.detach()], dim=1)

        z_where_shift = \
            (2. / self.args.arch.num_cell) * \
            (self.offset + 0.5 + torch.tanh(z_where_shift)) - 1.

        scale, ratio = z_where_scale.chunk(2, 1)
        scale = scale.sigmoid()
        ratio = torch.exp(ratio)
        ratio_sqrt = ratio.sqrt()
        z_where_scale = torch.cat([scale / ratio_sqrt, scale * ratio_sqrt], dim=1)
        # z_where: (bs, dim, num_cell, num_cell)
        z_where = torch.cat([z_where_scale, z_where_shift], dim=1)

        if phase_use_mode:
            z_depth = p_depth_mean
            z_what = p_what_mean
        else:
            z_depth = dist.Normal(p_depth_mean, p_depth_std).rsample()
            z_what = dist.Normal(p_what_mean, p_what_std).rsample()

        z_what_reshape = z_what.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_what_dim). \
            view(-1, self.args.z.z_what_dim, 1, 1)

        if self.args.data.inp_channel == 1 or not self.args.arch.phase_overlap:
            o = self.z_what_decoder_net(z_what_reshape)
            o = o.sigmoid()
            a = o.new_ones(o.size())
        elif self.args.arch.phase_overlap:
            o, a = self.z_what_decoder_net(z_what_reshape).split([self.args.data.inp_channel, 1], dim=1)
            o, a = o.sigmoid(), a.sigmoid()
        else:
            raise NotImplemented

        lv = [z_pres, z_where, z_depth, z_what, z_where_origin]
        pa = [o, a]

        return pa, lv


class StructDRAW(nn.Module):

    def __init__(self, args):
        super(StructDRAW, self).__init__()
        self.args = args

        self.p_global_decoder_net = StackMLP(
            self.args.z.z_global_dim,
            self.args.arch.mlp.p_global_decoder_filters,
            norm_act_final=True
        )

        rnn_enc_inp_dim = self.args.arch.img_enc_dim * 2 + \
                          self.args.arch.structdraw.rnn_decoder_hid_dim

        rnn_dec_inp_dim = self.args.arch.mlp.p_global_decoder_filters[-1] // \
                          (self.args.arch.num_cell ** 2)

        rnn_dec_inp_dim += self.args.arch.structdraw.hid_to_dec_filters[-1]

        self.rnn_enc = ConvLSTMCell(
            input_dim=rnn_enc_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_encoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.rnn_dec = ConvLSTMCell(
            input_dim=rnn_dec_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_decoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.p_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.mlp.p_global_encoder_filters,
            norm_act_final=False
        )

        self.q_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_encoder_hid_dim,
            self.args.arch.mlp.q_global_encoder_filters,
            norm_act_final=False
        )

        self.hid_to_dec_net = StackConvNorm(
            self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.structdraw.hid_to_dec_filters,
            self.args.arch.structdraw.hid_to_dec_kernel_sizes,
            self.args.arch.structdraw.hid_to_dec_strides,
            self.args.arch.structdraw.hid_to_dec_groups,
            norm_act_final=False
        )

        self.register_buffer('dec_step_0', torch.zeros(1, self.args.arch.structdraw.hid_to_dec_filters[-1],
                                                       self.args.arch.num_cell, self.args.arch.num_cell))

    def forward(self, x: torch.Tensor, phase_generation: bool = False,
                generation_from_step: Any = None, z_global_predefine: Any = None) -> Tuple:
        """
        :param x: (bs, dim, num_cell, num_cell) of (bs, dim, img_h, img_w)
        :return:
        """

        bs = x.size(0)

        h_enc, c_enc = self.rnn_enc.init_hidden(bs)
        h_dec, c_dec = self.rnn_dec.init_hidden(bs)

        p_global_mean_list = []
        p_global_std_list = []
        q_global_mean_list = []
        q_global_std_list = []
        z_global_list = []

        dec_step = self.dec_step_0.expand(bs, -1, -1, -1)

        for i in range(self.args.arch.draw_step):

            p_global_mean_step, p_global_std_step = \
                self.p_global_net(h_dec.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)
            p_global_std_step = F.softplus(p_global_std_step)

            if phase_generation or (generation_from_step is not None and i >= generation_from_step):

                q_global_mean_step = x.new_empty(bs, self.args.z.z_global_dim)
                q_global_std_step = x.new_empty(bs, self.args.z.z_global_dim)

                if z_global_predefine is None or z_global_predefine.size(1) <= i:
                    z_global_step = dist.Normal(p_global_mean_step, p_global_std_step).rsample()
                else:
                    z_global_step = z_global_predefine.view(bs, -1, self.args.z.z_global_dim)[:, i]

            else:

                if i == 0:
                    rnn_encoder_inp = torch.cat([x, x, h_dec], dim=1)
                else:
                    rnn_encoder_inp = torch.cat([x, x - dec_step, h_dec], dim=1)

                h_enc, c_enc = self.rnn_enc(rnn_encoder_inp, [h_enc, c_enc])

                q_global_mean_step, q_global_std_step = \
                    self.q_global_net(h_enc.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)

                q_global_std_step = F.softplus(q_global_std_step)
                z_global_step = dist.Normal(q_global_mean_step, q_global_std_step).rsample()

            rnn_decoder_inp = self.p_global_decoder_net(z_global_step). \
                reshape(bs, -1, self.args.arch.num_cell, self.args.arch.num_cell)

            rnn_decoder_inp = torch.cat([rnn_decoder_inp, dec_step], dim=1)

            h_dec, c_dec = self.rnn_dec(rnn_decoder_inp, [h_dec, c_dec])

            dec_step = dec_step + self.hid_to_dec_net(h_dec)

            # (bs, dim)
            p_global_mean_list.append(p_global_mean_step)
            p_global_std_list.append(p_global_std_step)
            q_global_mean_list.append(q_global_mean_step)
            q_global_std_list.append(q_global_std_step)
            z_global_list.append(z_global_step)

        global_dec = dec_step

        # (bs, steps, dim, 1, 1)
        p_global_mean_all = torch.stack(p_global_mean_list, 1)[:, :, :, None, None]
        p_global_std_all = torch.stack(p_global_std_list, 1)[:, :, :, None, None]
        q_global_mean_all = torch.stack(q_global_mean_list, 1)[:, :, :, None, None]
        q_global_std_all = torch.stack(q_global_std_list, 1)[:, :, :, None, None]
        z_global_all = torch.stack(z_global_list, 1)[:, :, :, None, None]

        pa = [global_dec]
        lv = [z_global_all]
        ss = [p_global_mean_all, p_global_std_all, q_global_mean_all, q_global_std_all]

        return pa, lv, ss


class BgEncoder(nn.Module):

    def __init__(self, args):
        super(BgEncoder, self).__init__()
        self.args = args

        self.p_bg_encoder = StackMLP(
            self.args.arch.img_enc_dim * self.args.arch.num_cell ** 2,
            self.args.arch.mlp.q_bg_encoder_filters,
            norm_act_final=False
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        """
        :param x: (bs, dim, img_h, img_w) or (bs, dim, num_cell, num_cell)
        :return:
        """
        bs = x.size(0)
        q_bg_mean, q_bg_std = self.p_bg_encoder(x.view(bs, -1)).chunk(2, 1)
        q_bg_mean = q_bg_mean.view(bs, -1, 1, 1)
        q_bg_std = q_bg_std.view(bs, -1, 1, 1)

        q_bg_std = F.softplus(q_bg_std)

        z_bg = dist.Normal(q_bg_mean, q_bg_std).rsample()

        lv = [z_bg]

        ss = [q_bg_mean, q_bg_std]

        return lv, ss


class BgGenerator(nn.Module):

    def __init__(self, args):
        super(BgGenerator, self).__init__()
        self.args = args

        inp_dim = self.args.z.z_global_dim * self.args.arch.draw_step

        self.p_bg_generator = StackMLP(
            inp_dim,
            self.args.arch.mlp.p_bg_generator_filters,
            norm_act_final=False
        )

    def forward(self, z_global_all: torch.Tensor, phase_use_mode: bool = False) -> Tuple:
        """
        :param x: (bs, step, dim, 1, 1)
        :return:
        """
        bs = z_global_all.size(0)

        bg_generator_inp = z_global_all

        p_bg_mean, p_bg_std = self.p_bg_generator(bg_generator_inp.view(bs, -1)).chunk(2, 1)
        p_bg_std = F.softplus(p_bg_std)

        p_bg_mean = p_bg_mean.view(bs, -1, 1, 1)
        p_bg_std = p_bg_std.view(bs, -1, 1, 1)

        if phase_use_mode:
            z_bg = p_bg_mean
        else:
            z_bg = dist.Normal(p_bg_mean, p_bg_std).rsample()

        lv = [z_bg]

        ss = [p_bg_mean, p_bg_std]

        return lv, ss


class BgDecoder(nn.Module):

    def __init__(self, args):
        super(BgDecoder, self).__init__()
        self.args = args

        self.p_bg_decoder = StackSubPixelNorm(
            self.args.z.z_bg_dim,
            self.args.arch.conv.p_bg_decoder_filters,
            self.args.arch.conv.p_bg_decoder_kernel_sizes,
            self.args.arch.conv.p_bg_decoder_upscales,
            self.args.arch.conv.p_bg_decoder_groups,
            norm_act_final=False
        )

    def forward(self, z_bg: torch.Tensor) -> List:
        """
        :param x: (bs, dim, 1, 1)
        :return:
        """
        bs = z_bg.size(0)

        bg = self.p_bg_decoder(z_bg).sigmoid()

        pa = [bg]

        return pa


class GNM(nn.Module):
    shortname = 'gnm'
    def __init__(self, args):
        super(GNM, self).__init__()
        self.args = args

        self.img_encoder = StackConvNorm(
            self.args.data.inp_channel,
            self.args.arch.conv.img_encoder_filters,
            self.args.arch.conv.img_encoder_kernel_sizes,
            self.args.arch.conv.img_encoder_strides,
            self.args.arch.conv.img_encoder_groups,
            norm_act_final=True
        )
        self.global_struct_draw = StructDRAW(self.args)
        self.p_z_given_x_or_g_net = LocalLatentDecoder(self.args)
        # Share latent decoder for p and q
        self.local_latent_sampler = LocalSampler(self.args)

        if self.args.arch.phase_background:
            self.p_bg_decoder = BgDecoder(self.args)
            self.p_bg_given_g_net = BgGenerator(self.args)
            self.q_bg_given_x_net = BgEncoder(self.args)

        self.register_buffer('aux_p_what_mean', torch.zeros(1))
        self.register_buffer('aux_p_what_std', torch.ones(1))
        self.register_buffer('aux_p_bg_mean', torch.zeros(1))
        self.register_buffer('aux_p_bg_std', torch.ones(1))
        self.register_buffer('aux_p_depth_mean', torch.zeros(1))
        self.register_buffer('aux_p_depth_std', torch.ones(1))
        self.register_buffer('aux_p_where_mean',
                             torch.tensor([self.args.const.scale_mean, self.args.const.ratio_mean, 0, 0])[None, :])
        # self.register_buffer('auxiliary_where_std', torch.ones(1))
        self.register_buffer('aux_p_where_std',
                             torch.tensor([self.args.const.scale_std, self.args.const.ratio_std,
                                           self.args.const.shift_std, self.args.const.shift_std])[None, :])
        self.register_buffer('aux_p_pres_probs', torch.tensor(self.args.train.p_pres_anneal_start_value))
        self.register_buffer('background', torch.zeros(1, self.args.data.inp_channel,
                                                       self.args.data.img_h, self.args.data.img_w))

    @property
    def aux_p_what(self):
        return dist.Normal(self.aux_p_what_mean, self.aux_p_what_std)

    @property
    def aux_p_bg(self):
        return dist.Normal(self.aux_p_bg_mean, self.aux_p_bg_std)

    @property
    def aux_p_depth(self):
        return dist.Normal(self.aux_p_depth_mean, self.aux_p_depth_std)

    @property
    def aux_p_where(self):
        return dist.Normal(self.aux_p_where_mean, self.aux_p_where_std)

    def forward(self, x: torch.Tensor, global_step) -> Tuple:
        self.args = hyperparam_anneal(self.args, global_step)
        bs = x.size(0)

        img_enc = self.img_encoder(x)
        if self.args.arch.phase_background:
            lv_q_bg, ss_q_bg = self.q_bg_given_x_net(img_enc)
            q_bg_mean, q_bg_std = ss_q_bg
        else:
            lv_q_bg = [self.background.new_zeros(1, 1)]
            q_bg_mean = self.background.new_zeros(1, 1)
            q_bg_std = self.background.new_ones(1, 1)
            ss_q_bg = [q_bg_mean, q_bg_std]

        q_bg = dist.Normal(q_bg_mean, q_bg_std)

        pa_g, lv_g, ss_g = self.global_struct_draw(img_enc)

        global_dec = pa_g[0]

        p_global_mean_all, p_global_std_all, q_global_mean_all, q_global_std_all = ss_g

        p_global_all = dist.Normal(p_global_mean_all, p_global_std_all)

        q_global_all = dist.Normal(q_global_mean_all, q_global_std_all)

        ss_p_z = self.p_z_given_x_or_g_net(global_dec)

        # (bs, dim, num_cell, num_cell)
        p_pres_logits, p_where_mean, p_where_std, p_depth_mean, \
        p_depth_std, p_what_mean, p_what_std = ss_p_z

        p_pres_given_g_probs_reshaped = torch.sigmoid(
            p_pres_logits.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )

        p_where_given_g = dist.Normal(
            p_where_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            p_where_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )
        p_depth_given_g = dist.Normal(
            p_depth_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            p_depth_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )
        p_what_given_g = dist.Normal(
            p_what_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            p_what_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )

        ss_q_z = self.p_z_given_x_or_g_net(img_enc, ss_p_z=ss_p_z)

        # (bs, dim, num_cell, num_cell)
        q_pres_logits, q_where_mean, q_where_std, q_depth_mean, \
        q_depth_std, q_what_mean, q_what_std = ss_q_z

        q_pres_given_x_and_g_probs_reshaped = torch.sigmoid(
            q_pres_logits.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )

        q_where_given_x_and_g = dist.Normal(
            q_where_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            q_where_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )
        q_depth_given_x_and_g = dist.Normal(
            q_depth_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            q_depth_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )
        q_what_given_x_and_g = dist.Normal(
            q_what_mean.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1),
            q_what_std.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        )

        if self.args.arch.phase_background:
            # lv_p_bg, ss_p_bg = self.ss_p_bg_given_g(lv_g)
            lv_p_bg, ss_p_bg = self.p_bg_given_g_net(lv_g[0], phase_use_mode=False)
            p_bg_mean, p_bg_std = ss_p_bg
        else:
            lv_p_bg = [self.background.new_zeros(1, 1)]
            p_bg_mean = self.background.new_zeros(1, 1)
            p_bg_std = self.background.new_ones(1, 1)

        p_bg = dist.Normal(p_bg_mean, p_bg_std)

        pa_recon, lv_z = self.lv_p_x_given_z_and_bg(ss_q_z, lv_q_bg)
        *pa_recon, patches, masks = pa_recon
        canvas = pa_recon[0]
        background = pa_recon[-1]

        z_pres, z_where, z_depth, z_what, z_where_origin = lv_z

        p_dists = [p_global_all, p_pres_given_g_probs_reshaped,
                   p_where_given_g, p_depth_given_g, p_what_given_g, p_bg]

        q_dists = [q_global_all, q_pres_given_x_and_g_probs_reshaped,
                   q_where_given_x_and_g, q_depth_given_x_and_g, q_what_given_x_and_g, q_bg]

        log_like, kl, log_imp = \
            self.elbo(x, p_dists, q_dists, lv_z, lv_g, lv_q_bg, pa_recon)

        self.log = {}

        if self.args.log.phase_log:
            pa_recon_from_q_g, _ = self.get_recon_from_q_g(global_dec=global_dec, lv_g=lv_g)

            z_pres_permute = z_pres.permute(0, 2, 3, 1)
            self.log = {
                'z_what': z_what.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_what_dim),
                'z_where_scale':
                    z_where.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)
                    [:, :self.args.z.z_where_scale_dim],
                'z_where_shift':
                    z_where.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)
                    [:, self.args.z.z_where_scale_dim:],
                'z_where_origin': z_where_origin.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_where_dim),
                'z_pres': z_pres_permute,
                'p_pres_probs': p_pres_given_g_probs_reshaped,
                'p_pres_logits': p_pres_logits,
                'p_what_std': p_what_std.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_what_dim)[z_pres_permute.view(-1) > 0.05],
                'p_what_mean': p_what_mean.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_what_dim)[z_pres_permute.view(-1) > 0.05],
                'p_where_scale_std':
                    p_where_std.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, :self.args.z.z_where_scale_dim],
                'p_where_scale_mean':
                    p_where_mean.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, :self.args.z.z_where_scale_dim],
                'p_where_shift_std':
                    p_where_std.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, self.args.z.z_where_scale_dim:],
                'p_where_shift_mean':
                    p_where_mean.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, self.args.z.z_where_scale_dim:],

                'q_pres_probs': q_pres_given_x_and_g_probs_reshaped,
                'q_pres_logits': q_pres_logits,
                'q_what_std': q_what_std.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_what_dim)[z_pres_permute.view(-1) > 0.05],
                'q_what_mean': q_what_mean.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_what_dim)[z_pres_permute.view(-1) > 0.05],
                'q_where_scale_std':
                    q_where_std.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, :self.args.z.z_where_scale_dim],
                'q_where_scale_mean':
                    q_where_mean.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, :self.args.z.z_where_scale_dim],
                'q_where_shift_std':
                    q_where_std.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, self.args.z.z_where_scale_dim:],
                'q_where_shift_mean':
                    q_where_mean.permute(0, 2, 3, 1).
                        reshape(-1, self.args.z.z_where_dim)[z_pres_permute.view(-1) > 0.05]
                    [:, self.args.z.z_where_scale_dim:],

                'z_depth': z_depth.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_depth_dim),
                'p_depth_std': p_depth_std.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_depth_dim)[z_pres_permute.view(-1) > 0.05],
                'p_depth_mean': p_depth_mean.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_depth_dim)[z_pres_permute.view(-1) > 0.05],
                'q_depth_std': q_depth_std.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_depth_dim)[z_pres_permute.view(-1) > 0.05],
                'q_depth_mean': q_depth_mean.permute(0, 2, 3, 1).
                    reshape(-1, self.args.z.z_depth_dim)[z_pres_permute.view(-1) > 0.05],
                'recon': pa_recon[0],
                'recon_from_q_g': pa_recon_from_q_g[0],
                'log_prob_x_given_g': dist.Normal(pa_recon_from_q_g[0], self.args.const.likelihood_sigma).
                    log_prob(x).flatten(start_dim=1).sum(1),
                'global_dec': global_dec,
            }
            z_global_all = lv_g[0]
            for i in range(self.args.arch.draw_step):
                self.log[f'z_global_step_{i}'] = z_global_all[:, i]
                self.log[f'q_global_mean_step_{i}'] = q_global_mean_all[:, i]
                self.log[f'q_global_std_step_{i}'] = q_global_std_all[:, i]
                self.log[f'p_global_mean_step_{i}'] = p_global_mean_all[:, i]
                self.log[f'p_global_std_step_{i}'] = p_global_std_all[:, i]
            if self.args.arch.phase_background:
                self.log['z_bg'] = lv_q_bg[0]
                self.log['p_bg_mean'] = p_bg_mean
                self.log['p_bg_std'] = p_bg_std
                self.log['q_bg_mean'] = q_bg_mean
                self.log['q_bg_std'] = q_bg_std
                self.log['recon_from_q_g_bg'] = pa_recon_from_q_g[-1]
                self.log['recon_from_q_g_fg'] = pa_recon_from_q_g[1]
                self.log['recon_from_q_g_alpha'] = pa_recon_from_q_g[2]
                self.log['recon_bg'] = pa_recon[-1]
                self.log['recon_fg'] = pa_recon[1]
                self.log['recon_alpha'] = pa_recon[2]

        ss = [ss_q_z, ss_q_bg, ss_g[2:]]
        aux_kl_pres, aux_kl_where, aux_kl_depth, aux_kl_what, aux_kl_bg, kl_pres, \
        kl_where, kl_depth, kl_what, kl_global_all, kl_bg = kl

        aux_kl_pres_raw = aux_kl_pres.mean(dim=0)
        aux_kl_where_raw = aux_kl_where.mean(dim=0)
        aux_kl_depth_raw = aux_kl_depth.mean(dim=0)
        aux_kl_what_raw = aux_kl_what.mean(dim=0)
        aux_kl_bg_raw = aux_kl_bg.mean(dim=0)
        kl_pres_raw = kl_pres.mean(dim=0)
        kl_where_raw = kl_where.mean(dim=0)
        kl_depth_raw = kl_depth.mean(dim=0)
        kl_what_raw = kl_what.mean(dim=0)
        kl_bg_raw = kl_bg.mean(dim=0)

        log_like = log_like.mean(dim=0)

        aux_kl_pres = aux_kl_pres_raw * self.args.train.beta_aux_pres
        aux_kl_where = aux_kl_where_raw * self.args.train.beta_aux_where
        aux_kl_depth = aux_kl_depth_raw * self.args.train.beta_aux_depth
        aux_kl_what = aux_kl_what_raw * self.args.train.beta_aux_what
        aux_kl_bg = aux_kl_bg_raw * self.args.train.beta_aux_bg
        kl_pres = kl_pres_raw * self.args.train.beta_pres
        kl_where = kl_where_raw * self.args.train.beta_where
        kl_depth = kl_depth_raw * self.args.train.beta_depth
        kl_what = kl_what_raw * self.args.train.beta_what
        kl_bg = kl_bg_raw * self.args.train.beta_bg

        kl_global_raw = kl_global_all.sum(dim=-1).mean(dim=0)
        kl_global = kl_global_raw * self.args.train.beta_global

        recon_loss = log_like
        kl = kl_pres + kl_where + kl_depth + kl_what + kl_bg + kl_global + \
             aux_kl_pres + aux_kl_where + aux_kl_depth + aux_kl_what + aux_kl_bg
        elbo = recon_loss - kl
        loss = - elbo

        bbox = visualize(x.cpu(),
                         self.log['z_pres'].view(bs, self.args.arch.num_cell ** 2, -1).cpu().detach(),
                         self.log['z_where_scale'].view(bs, self.args.arch.num_cell ** 2, -1).cpu().detach(),
                         self.log['z_where_shift'].view(bs, self.args.arch.num_cell ** 2, -1).cpu().detach(),
                         only_bbox=True, phase_only_display_pres=False)

        bbox = bbox.view(x.shape[0], -1, 3, self.args.data.img_h, self.args.data.img_w).sum(1).clamp(0.0, 1.0)
        # bbox_img = x.cpu().expand(-1, 3, -1, -1).contiguous()
        # bbox_img[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5] = \
        #     bbox[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5]
        ret = {
            'canvas': canvas,
            'canvas_with_bbox': bbox,
            'background': background,
            'steps': {
                'patch': patches,
                'mask': masks,
                'z_pres': z_pres.view(bs, self.args.arch.num_cell ** 2, -1)
            },
            'counts': torch.round(z_pres).flatten(1).sum(-1),
            'loss': loss,
            'elbo': elbo,
            'kl': kl,
            'rec_loss': recon_loss,
            'kl_pres': kl_pres,
            'kl_aux_pres': aux_kl_pres,
            'kl_where': kl_where,
            'kl_aux_where': aux_kl_where,
            'kl_what': kl_what,
            'kl_aux_what': aux_kl_what,
            'kl_depth': kl_depth,
            'kl_aux_depth': aux_kl_depth,
            'kl_bg': kl_bg,
            'kl_aux_bg': aux_kl_bg,
            'kl_global': kl_global
        }

        # return pa_recon, log_like, kl, log_imp, lv_z + lv_g + lv_q_bg, ss, self.log
        return ret

    def get_recon_from_q_g(
            self,
            img: torch.Tensor = None,
            global_dec: torch.Tensor = None,
            lv_g: List = None,
            phase_use_mode: bool = False
    ) -> Tuple:

        assert img is not None or (global_dec is not None and lv_g is not None), "Provide either image or p_l_given_g"
        if img is not None:
            img_enc = self.img_encoder(img)
            pa_g, lv_g, ss_g = self.global_struct_draw(img_enc)

            global_dec = pa_g[0]

        if self.args.arch.phase_background:
            lv_p_bg, _ = self.p_bg_given_g_net(lv_g[0], phase_use_mode=phase_use_mode)
        else:
            lv_p_bg = [self.background.new_zeros(1, 1)]

        ss_z = self.p_z_given_x_or_g_net(global_dec)

        pa, lv = self.lv_p_x_given_z_and_bg(ss_z, lv_p_bg, phase_use_mode=phase_use_mode)

        lv = lv + lv_p_bg

        return pa, lv

    def sample(self, phase_use_mode: bool = False):

        dummy_x = self.aux_p_pres_probs.new_zeros([self.args.log.num_sample, 1])
        pa_g, lv_g, _ = self.global_struct_draw(dummy_x, phase_generation=True)

        if self.args.arch.phase_background:
            lv_p_bg, _ = self.p_bg_given_g_net(lv_g[0], phase_use_mode=phase_use_mode)
        else:
            lv_p_bg = [self.background.new_zeros(1, 1)]

        global_dec = pa_g[0]

        ss_z = self.p_z_given_x_or_g_net(global_dec)

        pa, lv = self.lv_p_x_given_z_and_bg(ss_z, lv_p_bg, phase_use_mode=phase_use_mode)

        z_global_all = lv_g[0]
        lv_g_z = [z_global_all] + lv

        return pa, lv_g_z, ss_z

    def elbo(self,
             x: torch.Tensor,
             p_dists: List,
             q_dists: List,
             lv_z: List,
             lv_g: List,
             lv_bg: List,
             pa_recon: List) -> Tuple:

        bs = x.size(0)

        p_global_all, p_pres_given_g_probs_reshaped, \
        p_where_given_g, p_depth_given_g, p_what_given_g, p_bg = p_dists

        q_global_all, q_pres_given_x_and_g_probs_reshaped, \
        q_where_given_x_and_g, q_depth_given_x_and_g, q_what_given_x_and_g, q_bg = q_dists

        y, y_nobg, alpha_map, bg = pa_recon

        if self.args.log.phase_nll:
            # (bs, dim, num_cell, num_cell)
            z_pres, _, z_depth, z_what, z_where_origin = lv_z
            # (bs * num_cell * num_cell, dim)
            z_pres_reshape = z_pres.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_pres_dim)
            z_depth_reshape = z_depth.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_depth_dim)
            z_what_reshape = z_what.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_what_dim)
            z_where_origin_reshape = z_where_origin.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_where_dim)
            # (bs, dim, 1, 1)
            z_bg = lv_bg[0]
            # (bs, step, dim, 1, 1)
            z_g = lv_g[0]
        else:
            z_pres, _, _, _, z_where_origin = lv_z

            z_pres_reshape = z_pres.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_pres_dim)

        if self.args.train.p_pres_anneal_end_step != 0:
            self.aux_p_pres_probs = linear_schedule_tensor(
                self.args.train.global_step,
                self.args.train.p_pres_anneal_start_step,
                self.args.train.p_pres_anneal_end_step,
                self.args.train.p_pres_anneal_start_value,
                self.args.train.p_pres_anneal_end_value,
                self.aux_p_pres_probs.device
            )

        if self.args.train.aux_p_scale_anneal_end_step != 0:
            aux_p_scale_mean = linear_schedule_tensor(
                self.args.train.global_step,
                self.args.train.aux_p_scale_anneal_start_step,
                self.args.train.aux_p_scale_anneal_end_step,
                self.args.train.aux_p_scale_anneal_start_value,
                self.args.train.aux_p_scale_anneal_end_value,
                self.aux_p_where_mean.device
            )
            self.aux_p_where_mean[:, 0] = aux_p_scale_mean

        auxiliary_prior_z_pres_probs = self.aux_p_pres_probs[None][None, :].expand(
            bs * self.args.arch.num_cell ** 2, -1)

        aux_kl_pres = kl_divergence_bern_bern(q_pres_given_x_and_g_probs_reshaped, auxiliary_prior_z_pres_probs)
        aux_kl_where = dist.kl_divergence(q_where_given_x_and_g, self.aux_p_where) * z_pres_reshape.clamp(min=1e-5)
        aux_kl_depth = dist.kl_divergence(q_depth_given_x_and_g, self.aux_p_depth) * z_pres_reshape.clamp(min=1e-5)
        aux_kl_what = dist.kl_divergence(q_what_given_x_and_g, self.aux_p_what) * z_pres_reshape.clamp(min=1e-5)

        kl_pres = kl_divergence_bern_bern(q_pres_given_x_and_g_probs_reshaped, p_pres_given_g_probs_reshaped)

        kl_where = dist.kl_divergence(q_where_given_x_and_g, p_where_given_g)
        kl_depth = dist.kl_divergence(q_depth_given_x_and_g, p_depth_given_g)
        kl_what = dist.kl_divergence(q_what_given_x_and_g, p_what_given_g)

        kl_global_all = dist.kl_divergence(q_global_all, p_global_all)

        if self.args.arch.phase_background:
            kl_bg = dist.kl_divergence(q_bg, p_bg)
            aux_kl_bg = dist.kl_divergence(q_bg, self.aux_p_bg)
        else:
            kl_bg = self.background.new_zeros(bs, 1)
            aux_kl_bg = self.background.new_zeros(bs, 1)

        log_like = dist.Normal(y, self.args.const.likelihood_sigma).log_prob(x)

        log_imp_list = []
        if self.args.log.phase_nll:
            log_pres_prior = z_pres_reshape * torch.log(p_pres_given_g_probs_reshaped + self.args.const.eps) + \
                             (1 - z_pres_reshape) * torch.log(1 - p_pres_given_g_probs_reshaped + self.args.const.eps)
            log_pres_pos = z_pres_reshape * torch.log(q_pres_given_x_and_g_probs_reshaped + self.args.const.eps) + \
                           (1 - z_pres_reshape) * torch.log(
                1 - q_pres_given_x_and_g_probs_reshaped + self.args.const.eps)

            log_imp_pres = log_pres_prior - log_pres_pos

            log_imp_depth = p_depth_given_g.log_prob(z_depth_reshape) - \
                            q_depth_given_x_and_g.log_prob(z_depth_reshape)

            log_imp_what = p_what_given_g.log_prob(z_what_reshape) - \
                           q_what_given_x_and_g.log_prob(z_what_reshape)

            log_imp_where = p_where_given_g.log_prob(z_where_origin_reshape) - \
                            q_where_given_x_and_g.log_prob(z_where_origin_reshape)

            if self.args.arch.phase_background:
                log_imp_bg = p_bg.log_prob(z_bg) - q_bg.log_prob(z_bg)
            else:
                log_imp_bg = x.new_zeros(bs, 1)

            log_imp_g = p_global_all.log_prob(z_g) - q_global_all.log_prob(z_g)

            log_imp_list = [
                log_imp_pres.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(1),
                log_imp_depth.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                    1),
                log_imp_what.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(1),
                log_imp_where.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                    1),
                log_imp_bg.flatten(start_dim=1).sum(1),
                log_imp_g.flatten(start_dim=1).sum(1),
            ]

        return log_like.flatten(start_dim=1).sum(1), \
               [
                   aux_kl_pres.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                       -1),
                   aux_kl_where.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                       -1),
                   aux_kl_depth.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                       -1),
                   aux_kl_what.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(
                       -1),
                   aux_kl_bg.flatten(start_dim=1).sum(-1),
                   kl_pres.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(-1),
                   kl_where.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(-1),
                   kl_depth.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(-1),
                   kl_what.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1).flatten(start_dim=1).sum(-1),
                   kl_global_all.flatten(start_dim=2).sum(-1),
                   kl_bg.flatten(start_dim=1).sum(-1)
               ], log_imp_list

    # def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     :param x: (bs, inp_channel, img_h, img_w)
    #     :return: img_enc: (bs, dim, num_cell, num_cell)
    #     """
    #
    #     img_enc = self.img_encoder(x)
    #
    #     return img_enc

    # def ss_p_z_given_g(self, global_dec: torch.Tensor) -> List:
    #     """
    #     :param x: sample of z_global variable (bs, dim, 1, 1)
    #     :return:
    #     """
    #     ss_z = self.p_z_given_g_net(global_dec)
    #
    #     return ss_z

    # def ss_q_z_given_x(self, img_enc: torch.Tensor, global_dec: torch.Tensor, ss_p_z: List) -> List:
    #     """
    #     :param x: sample of z_global variable (bs, dim, 1, 1)
    #     :return:
    #     """
    #     ss_z = self.p_z_given_x_or_g_net(img_enc, ss_p_z=ss_p_z)
    #
    #     return ss_z

    # def ss_q_bg_given_x(self, x: torch.Tensor) -> Tuple:
    #     """
    #     :param x: (bs, dim, img_h, img_w)
    #     :return:
    #     """
    #     lv_q_bg, ss_q_bg = self.q_bg_given_x_net(x)
    #
    #     return lv_q_bg, ss_q_bg

    # def ss_p_bg_given_g(self, lv_g: List, phase_use_mode: bool = False) -> Tuple:
    #     """
    #     :param x: (bs, dim, img_h, img_w)
    #     :return:
    #     """
    #     z_global_all = lv_g[0]
    #     lv_p_bg, ss_p_bg = self.p_bg_given_g_net(z_global_all, phase_use_mode=phase_use_mode)
    #
    #     return lv_p_bg, ss_p_bg

    def lv_p_x_given_z_and_bg(self, ss: List, lv_bg: List, phase_use_mode: bool = False) -> Tuple:
        """
        :param z: (bs, z_what_dim)
        :return:
        """
        # x: (bs, inp_channel, img_h, img_w)
        pa, lv_z = self.local_latent_sampler(ss, phase_use_mode=phase_use_mode)

        o_att, a_att, *_ = pa
        z_pres, z_where, z_depth, *_ = lv_z

        if self.args.arch.phase_background:
            z_bg = lv_bg[0]
            pa_bg = self.p_bg_decoder(z_bg)
            y_bg = pa_bg[0]
        else:
            # pa_bg = [self.background.expand(lv_z[0].size(0), -1, -1, -1)]
            y_bg = self.background.expand(lv_z[0].size(0), -1, -1, -1)

        # pa = pa + pa_bg

        y, y_fg, alpha_map, patches, masks = self.render(o_att, a_att, y_bg, z_pres, z_where, z_depth)

        return [y, y_fg, alpha_map, y_bg, patches, masks], lv_z

    # def pa_bg_given_z_bg(self, lv_bg: List) -> List:
    #     """
    #     :param lv_bg[0]: (bs, z_bg_dim, 1, 1)
    #     :return:
    #     """
    #     z_bg = lv_bg[0]
    #     pa = self.p_bg_decoder(z_bg)
    #
    #     return pa

    def render(self, o_att, a_att, bg, z_pres, z_where, z_depth) -> List:
        """
        :param pa: variables with size (bs, dim, num_cell, num_cell)
        :param lv_z: o and a with size (bs * num_cell * num_cell, dim)
        :return:
        """

        bs = z_pres.size(0)

        z_pres = z_pres.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        z_where = z_where.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)
        z_depth = z_depth.permute(0, 2, 3, 1).reshape(bs * self.args.arch.num_cell ** 2, -1)

        if self.args.arch.phase_overlap == True:
            if self.args.train.phase_bg_alpha_curriculum and \
                    self.args.train.bg_alpha_curriculum_period[0] < self.args.train.global_step < \
                    self.args.train.bg_alpha_curriculum_period[1]:
                z_pres = z_pres.clamp(max=0.99)
            a_att_hat = a_att * z_pres.view(-1, 1, 1, 1)
            y_att = a_att_hat * o_att

            # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 3, img_h, img_w)
            y_att_full_res = spatial_transform(y_att, z_where,
                                               (bs * self.args.arch.num_cell ** 2, self.args.data.inp_channel,
                                                self.args.data.img_h, self.args.data.img_w),
                                               inverse=True).view(-1, self.args.arch.num_cell * self.args.arch.num_cell,
                                                                  self.args.data.inp_channel, self.args.data.img_h,
                                                                  self.args.data.img_w)
            o_att_full_res = spatial_transform(o_att, z_where,
                                               (bs * self.args.arch.num_cell ** 2, self.args.data.inp_channel,
                                                self.args.data.img_h, self.args.data.img_w),
                                               inverse=True).view(-1, self.args.arch.num_cell * self.args.arch.num_cell,
                                                                  self.args.data.inp_channel, self.args.data.img_h,
                                                                  self.args.data.img_w)

            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, glimpse_size, glimpse_size)
            importance_map = a_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, img_h, img_w)
            importance_map_full_res = spatial_transform(importance_map, z_where,
                                                        (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1,
                                                         self.args.data.img_h, self.args.data.img_w),
                                                        inverse=True)
            # # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 1, img_h, img_w)
            importance_map_full_res = \
                importance_map_full_res.view(-1, self.args.arch.num_cell * self.args.arch.num_cell, 1,
                                             self.args.data.img_h,
                                             self.args.data.img_w)
            importance_map_full_res_norm = importance_map_full_res / \
                                           (importance_map_full_res.sum(dim=1, keepdim=True) + self.args.const.eps)

            # (bs, 3, img_h, img_w)
            y_nobg = (y_att_full_res * importance_map_full_res_norm).sum(dim=1)

            # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 1, img_h, img_w)
            a_att_hat_full_res = spatial_transform(
                a_att_hat, z_where,
                (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, self.args.data.img_h,
                 self.args.data.img_w),
                inverse=True
            ).view(-1, self.args.arch.num_cell * self.args.arch.num_cell, 1, self.args.data.img_h,
                   self.args.data.img_w)
            alpha_map = a_att_hat_full_res.sum(dim=1)
            # (bs, 1, img_h, img_w)
            alpha_map = alpha_map + (
                    alpha_map.clamp(self.args.const.eps, 1 - self.args.const.eps) - alpha_map).detach()

            if self.args.train.phase_bg_alpha_curriculum:
                if self.args.train.bg_alpha_curriculum_period[0] < self.args.train.global_step < \
                        self.args.train.bg_alpha_curriculum_period[1]:
                    alpha_map = alpha_map.new_ones(alpha_map.size()) * self.args.train.bg_alpha_curriculum_value
                    # y_nobg = alpha_map * y_nobg
            y = y_nobg + (1. - alpha_map) * bg
        else:
            y_att = a_att * o_att

            o_att_full_res = spatial_transform(o_att, z_where,
                                               (bs * self.args.arch.num_cell ** 2, self.args.data.inp_channel,
                                                self.args.data.img_h, self.args.data.img_w),
                                               inverse=True).view(-1, self.args.arch.num_cell * self.args.arch.num_cell,
                                                                  self.args.data.inp_channel, self.args.data.img_h,
                                                                  self.args.data.img_w)
            a_att_hat_full_res = spatial_transform(
                a_att * z_pres.view(bs * self.args.arch.num_cell ** 2, 1, 1, 1), z_where,
                (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, self.args.data.img_h,
                 self.args.data.img_w),
                inverse=True
            ).view(-1, self.args.arch.num_cell * self.args.arch.num_cell, 1, self.args.data.img_h,
                   self.args.data.img_w)

            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 3, img_h, img_w)
            y_att_full_res = spatial_transform(
                y_att, z_where,
                (bs * self.args.arch.num_cell ** 2, self.args.data.inp_channel, self.args.data.img_h,
                 self.args.data.img_w),
                inverse=True
            )
            y = (y_att_full_res * z_pres.view(bs * self.args.arch.num_cell ** 2, 1, 1, 1)). \
                view(bs, -1, self.args.data.inp_channel, self.args.data.img_h, self.args.data.img_w).sum(dim=1)
            y_nobg = y
            alpha_map = y.new_ones(y.size(0), 1, y.size(2), y.size(3))

        return y, y_nobg, alpha_map, o_att_full_res, a_att_hat_full_res


import torch
import torch.nn.functional as F
import os

border_width = 3

rbox = torch.zeros(3, 42, 42)
rbox[0, :border_width, :] = 1
rbox[0, -border_width:, :] = 1
rbox[0, :, :border_width] = 1
rbox[0, :, -border_width:] = 1
rbox = rbox.view(1, 3, 42, 42)

gbox = torch.zeros(3, 42, 42)
gbox[1, :border_width, :] = 1
gbox[1, -border_width:, :] = 1
gbox[1, :, :border_width] = 1
gbox[1, :, -border_width:] = 1
gbox = gbox.view(1, 3, 42, 42)

wbox = torch.zeros(3, 42, 42)
wbox[:, :border_width, :] = 1
wbox[:, -border_width:, :] = 1
wbox[:, :, :border_width] = 1
wbox[:, :, -border_width:] = 1
wbox = wbox.view(1, 3, 42, 42)


def visualize(x, z_pres, z_where_scale, z_where_shift, only_bbox=False, phase_only_display_pres=True):
    """
        x: (bs, 3, img_h, img_w)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    bs, _, img_h, img_w = x.size()
    z_pres = z_pres.view(-1, 1, 1, 1)
    num_obj = z_pres.size(0) // bs
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    if phase_only_display_pres:
        bbox = spatial_transform(z_pres * gbox,
                                 torch.cat((z_scale, z_shift), dim=1),
                                 torch.Size([bs * num_obj, 3, img_h, img_w]),
                                 inverse=True)
    else:
        bbox = spatial_transform(z_pres * gbox + (1 - z_pres) * rbox,
                                 torch.cat((z_scale, z_shift), dim=1),
                                 torch.Size([bs * num_obj, 3, img_h, img_w]),
                                 inverse=True)

    if not only_bbox:
        bbox = (bbox + torch.stack(num_obj * (x,), dim=1).view(-1, 3, img_h, img_w)).clamp(0.0, 1.0)
    return bbox

import copy
def hyperparam_anneal(args, global_step):
    # args = copy.deepcopy(args)
    # pprint.pprint(args)
    args.train['global_step'] = global_step

    if args.train.beta_aux_pres_anneal_end_step == 0:
        args.train['beta_aux_pres'] = args.train.beta_aux_pres_anneal_start_value
    else:
        args.train['beta_aux_pres'] = linear_schedule(
            global_step,
            args.train.beta_aux_pres_anneal_start_step,
            args.train.beta_aux_pres_anneal_end_step,
            args.train.beta_aux_pres_anneal_start_value,
            args.train.beta_aux_pres_anneal_end_value
        )

    if args.train.beta_aux_where_anneal_end_step == 0:
        args.train['beta_aux_where'] = args.train.beta_aux_where_anneal_start_value
    else:
        args.train['beta_aux_where'] = linear_schedule(
            global_step,
            args.train.beta_aux_where_anneal_start_step,
            args.train.beta_aux_where_anneal_end_step,
            args.train.beta_aux_where_anneal_start_value,
            args.train.beta_aux_where_anneal_end_value
        )

    if args.train.beta_aux_what_anneal_end_step == 0:
        args.train['beta_aux_what'] = args.train.beta_aux_what_anneal_start_value
    else:
        args.train['beta_aux_what'] = linear_schedule(
            global_step,
            args.train.beta_aux_what_anneal_start_step,
            args.train.beta_aux_what_anneal_end_step,
            args.train.beta_aux_what_anneal_start_value,
            args.train.beta_aux_what_anneal_end_value
        )

    if args.train.beta_aux_depth_anneal_end_step == 0:
        args.train['beta_aux_depth'] = args.train.beta_aux_depth_anneal_start_value
    else:
        args.train['beta_aux_depth'] = linear_schedule(
            global_step,
            args.train.beta_aux_depth_anneal_start_step,
            args.train.beta_aux_depth_anneal_end_step,
            args.train.beta_aux_depth_anneal_start_value,
            args.train.beta_aux_depth_anneal_end_value
        )

    if args.train.beta_aux_global_anneal_end_step == 0:
        args.train['beta_aux_global'] = args.train.beta_aux_global_anneal_start_value
    else:
        args.train['beta_aux_global'] = linear_schedule(
            global_step,
            args.train.beta_aux_global_anneal_start_step,
            args.train.beta_aux_global_anneal_end_step,
            args.train.beta_aux_global_anneal_start_value,
            args.train.beta_aux_global_anneal_end_value
        )

    if args.train.beta_aux_bg_anneal_end_step == 0:
        args.train['beta_aux_bg'] = args.train.beta_aux_bg_anneal_start_value
    else:
        args.train['beta_aux_bg'] = linear_schedule(
            global_step,
            args.train.beta_aux_bg_anneal_start_step,
            args.train.beta_aux_bg_anneal_end_step,
            args.train.beta_aux_bg_anneal_start_value,
            args.train.beta_aux_bg_anneal_end_value
        )

    ########################### split here ###########################
    if args.train.beta_pres_anneal_end_step == 0:
        args.train['beta_pres'] = args.train.beta_pres_anneal_start_value
    else:
        args.train['beta_pres'] = linear_schedule(
            global_step,
            args.train.beta_pres_anneal_start_step,
            args.train.beta_pres_anneal_end_step,
            args.train.beta_pres_anneal_start_value,
            args.train.beta_pres_anneal_end_value
        )

    if args.train.beta_where_anneal_end_step == 0:
        args.train['beta_where'] = args.train.beta_where_anneal_start_value
    else:
        args.train['beta_where'] = linear_schedule(
            global_step,
            args.train.beta_where_anneal_start_step,
            args.train.beta_where_anneal_end_step,
            args.train.beta_where_anneal_start_value,
            args.train.beta_where_anneal_end_value
        )

    if args.train.beta_what_anneal_end_step == 0:
        args.train['beta_what'] = args.train.beta_what_anneal_start_value
    else:
        args.train['beta_what'] = linear_schedule(
            global_step,
            args.train.beta_what_anneal_start_step,
            args.train.beta_what_anneal_end_step,
            args.train.beta_what_anneal_start_value,
            args.train.beta_what_anneal_end_value
        )

    if args.train.beta_depth_anneal_end_step == 0:
        args.train['beta_depth'] = args.train.beta_depth_anneal_start_value
    else:
        args.train['beta_depth'] = linear_schedule(
            global_step,
            args.train.beta_depth_anneal_start_step,
            args.train.beta_depth_anneal_end_step,
            args.train.beta_depth_anneal_start_value,
            args.train.beta_depth_anneal_end_value
        )

    if args.train.beta_global_anneal_end_step == 0:
        args.train['beta_global'] = args.train.beta_global_anneal_start_value
    else:
        args.train['beta_global'] = linear_schedule(
            global_step,
            args.train.beta_global_anneal_start_step,
            args.train.beta_global_anneal_end_step,
            args.train.beta_global_anneal_start_value,
            args.train.beta_global_anneal_end_value
        )

    if args.train.tau_pres_anneal_end_step == 0:
        args.train['tau_pres'] = args.train.tau_pres_anneal_start_value
    else:
        args.train['tau_pres'] = linear_schedule(
            global_step,
            args.train.tau_pres_anneal_start_step,
            args.train.tau_pres_anneal_end_step,
            args.train.tau_pres_anneal_start_value,
            args.train.tau_pres_anneal_end_value
        )

    if args.train.beta_bg_anneal_end_step == 0:
        args.train['beta_bg'] = args.train.beta_bg_anneal_start_value
    else:
        args.train['beta_bg'] = linear_schedule(
            global_step,
            args.train.beta_bg_anneal_start_step,
            args.train.beta_bg_anneal_end_step,
            args.train.beta_bg_anneal_start_value,
            args.train.beta_bg_anneal_end_value
        )
    # print(args.train)
    # import ipdb; ipdb.set_trace()
    # print(args.train.global_step)
    return args

def linear_schedule(step, start_step, end_step, start_value, end_value):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = start_value + slope * (step - start_step)
    elif step >= end_step:
        x = end_value
    else:
        x = start_value
    return x


CONFIG_YAML = '''exp_name: ''
data_dir: ''
summary_dir: ''
model_dir: ''
last_ckpt: ''
data:
  img_w: 128
  img_h: 128
  inp_channel: 3
  blender_dir_list_train: []
  blender_dir_list_test: []
  dataset: 'mnist'
z:
  z_global_dim: 32
  z_what_dim: 64
  z_where_scale_dim: 2
  z_where_shift_dim: 2
  z_where_dim: 4
  z_pres_dim: 1
  z_depth_dim: 1
  z_local_dim: 64
  z_bg_dim: 10

arch:
  glimpse_size: 64
  num_cell: 4
  phase_overlap: True
  phase_background: True
  img_enc_dim: 128
  p_global_decoder_type: 'MLP'
  draw_step: 4
  phase_graph_net_on_global_decoder: False
  phase_graph_net_on_global_encoder: False

  conv:
    img_encoder_filters: [16, 16, 32, 32, 64, 64, 128, 128, 128]
    img_encoder_groups: [1, 1, 1, 1, 1, 1, 1, 1, 1]
    img_encoder_strides: [2, 1, 2, 1, 2, 1, 2, 1, 2]
    img_encoder_kernel_sizes: [4, 3, 4, 3, 4, 3, 4, 3, 4]

    p_what_decoder_filters: [128, 64, 32, 16, 8, 4]
    p_what_decoder_kernel_sizes: [3, 3, 3, 3, 3, 3]
    p_what_decoder_upscales: [2, 2, 2, 2, 2, 2]
    p_what_decoder_groups: [1, 1, 1, 1, 1, 1]

    p_bg_decoder_filters: [128, 64, 32, 16, 8, 3]
    p_bg_decoder_kernel_sizes: [1, 1, 1, 1, 1, 3]
    p_bg_decoder_upscales: [4, 2, 4, 2, 2, 1]
    p_bg_decoder_groups: [1, 1, 1, 1, 1, 1]

  deconv:
    p_global_decoder_filters: [128, 128, 128]
    p_global_decoder_kernel_sizes: [1, 1, 1]
    p_global_decoder_upscales: [2, 1, 2]
    p_global_decoder_groups: [1, 1, 1]

  mlp:
    p_global_decoder_filters: [512, 1024, 2048]
    q_global_encoder_filters: [512, 512, 64]
    p_global_encoder_filters: [512, 512, 64]
    p_bg_generator_filters: [128, 64, 20]
    q_bg_encoder_filters: [512, 256, 20]

  pwdw:
    pwdw_filters: [128, 128]
    pwdw_kernel_sizes: [1, 1]
    pwdw_strides: [1, 1]
    pwdw_groups: [1, 1]

  structdraw:
    kernel_size: 1
    rnn_decoder_hid_dim: 128
    rnn_encoder_hid_dim: 128
    hid_to_dec_filters: [128]
    hid_to_dec_kernel_sizes: [3]
    hid_to_dec_strides: [1]
    hid_to_dec_groups: [1]

log:
  num_summary_img: 15
  num_img_per_row: 5
  save_epoch_freq: 10
  print_step_freq: 2000
  num_sample: 50
  compute_nll_freq: 20
  phase_nll: False
  nll_num_sample: 30
  phase_log: True

const:
  pres_logit_scale: 8.8
  scale_mean: -1.5
  scale_std: 0.1
  ratio_mean: 0
  ratio_std: 0.3
  shift_std: 1
  eps: 0.000000000000001
  likelihood_sigma: 0.2
  bg_likelihood_sigma: 0.3

train:
  start_epoch: 0
  epoch: 600
  batch_size: 32
  lr: 0.0001
  cp: 1.0

  beta_global_anneal_start_step: 0
  beta_global_anneal_end_step: 100000
  beta_global_anneal_start_value: 0.
  beta_global_anneal_end_value: 1.

  beta_pres_anneal_start_step: 0
  beta_pres_anneal_end_step: 0
  beta_pres_anneal_start_value: 1.
  beta_pres_anneal_end_value: 0.

  beta_where_anneal_start_step: 0
  beta_where_anneal_end_step: 0
  beta_where_anneal_start_value: 1.
  beta_where_anneal_end_value: 0.

  beta_what_anneal_start_step: 0
  beta_what_anneal_end_step: 0
  beta_what_anneal_start_value: 1.
  beta_what_anneal_end_value: 0.

  beta_depth_anneal_start_step: 0
  beta_depth_anneal_end_step: 0
  beta_depth_anneal_start_value: 1.
  beta_depth_anneal_end_value: 0.

  beta_bg_anneal_start_step: 1000
  beta_bg_anneal_end_step: 0
  beta_bg_anneal_start_value: 1.
  beta_bg_anneal_end_value: 0.

  beta_aux_pres_anneal_start_step: 1000
  beta_aux_pres_anneal_end_step: 0
  beta_aux_pres_anneal_start_value: 1.
  beta_aux_pres_anneal_end_value: 0.

  beta_aux_where_anneal_start_step: 0
  beta_aux_where_anneal_end_step: 500
  beta_aux_where_anneal_start_value: 10.
  beta_aux_where_anneal_end_value: 1.

  beta_aux_what_anneal_start_step: 1000
  beta_aux_what_anneal_end_step: 0
  beta_aux_what_anneal_start_value: 1.
  beta_aux_what_anneal_end_value: 0.

  beta_aux_depth_anneal_start_step: 1000
  beta_aux_depth_anneal_end_step: 0
  beta_aux_depth_anneal_start_value: 1.
  beta_aux_depth_anneal_end_value: 0.

  beta_aux_global_anneal_start_step: 0
  beta_aux_global_anneal_end_step: 100000
  beta_aux_global_anneal_start_value: 0.
  beta_aux_global_anneal_end_value: 1.

  beta_aux_bg_anneal_start_step: 0
  beta_aux_bg_anneal_end_step: 50000
  beta_aux_bg_anneal_start_value: 50.
  beta_aux_bg_anneal_end_value: 1.

  tau_pres_anneal_start_step: 1000
  tau_pres_anneal_end_step: 20000
  tau_pres_anneal_start_value: 1.
  tau_pres_anneal_end_value: 0.5
  tau_pres: 1.

  p_pres_anneal_start_step: 0
  p_pres_anneal_end_step: 4000
  p_pres_anneal_start_value: 0.1
  p_pres_anneal_end_value: 0.001

  aux_p_scale_anneal_start_step: 0
  aux_p_scale_anneal_end_step: 0
  aux_p_scale_anneal_start_value: -1.5
  aux_p_scale_anneal_end_value: -1.5

  phase_bg_alpha_curriculum: True
  bg_alpha_curriculum_period: [0, 500]
  bg_alpha_curriculum_value: 0.9

  seed: 666
'''
import yaml
import io
from prodict import Prodict
arrow_args = Prodict.from_dict(yaml.safe_load(io.StringIO(CONFIG_YAML)))
import pprint
pprint.pprint(arrow_args)
