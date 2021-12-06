"""
Code adjusted from https://github.com/pemami4911/EfficientMORL

"Efficient Iterative Amortized Inference for Learning Symmetric and Disentangled Multi-Object Representations"
Patrick Emami, Pan He, Sanjay Ranka, Anand Rangarajan
ICML 2021
http://proceedings.mlr.press/v139/emami21a.html
"""

from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor

from ool.picture.models.thirdparty.emorl.utils import init_weights, mvn, std_mvn, GECO, gmm_negativeloglikelihood, \
    gaussian_negativeloglikelihood


class ImageDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask
    """

    # @net.capture
    def __init__(self, input_size, z_size, image_decoder, K, batch_size):
        super(ImageDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        output_size = 4
        small_grid_size = 6
        # Strides (2,2) with padding --- goes down to (8,8). From Slot Attention paper
        if image_decoder == 'big':
            self.decode = nn.Sequential(
                nn.ConvTranspose2d(z_size, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 1, 2, output_padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, output_size, 3, 1, 1, output_padding=0)
            )
            self.z_grid_shape = (small_grid_size, small_grid_size)
            self.positional_embedding = slot_attention_create_positional_embedding(small_grid_size, small_grid_size,
                                                                                  K * batch_size)
        elif image_decoder == 'iodine':
            self.decode = nn.Sequential(
                nn.Conv2d(z_size, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, output_size, 3, 1)
            )
            self.z_grid_shape = (self.h + 10, self.w + 10)
            self.positional_embedding = slot_attention_create_positional_embedding(
                self.z_grid_shape[0], self.z_grid_shape[1], K * batch_size)

        elif image_decoder == 'small':
            self.decode = nn.Sequential(
                nn.Conv2d(z_size, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, output_size, 3, 1, 1)
            )
            self.z_grid_shape = (self.h + 6, self.w + 6)
            self.positional_embedding = slot_attention_create_positional_embedding(
                self.z_grid_shape[0], self.z_grid_shape[1], K * batch_size)
        self.pos_embed_projection = nn.Linear(4, z_size)

    def forward(self, z):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, self.z_grid_shape[0], self.z_grid_shape[1])

        pos_embed = self.positional_embedding.to(z.device)  # [N,H,W,4]
        pos_embed = self.pos_embed_projection(pos_embed)  # [N,H,W,64]
        pos_embed = pos_embed.permute(0, 3, 1, 2).contiguous()

        z_b = z_b + pos_embed
        out = self.decode(z_b)  # [batch_size * K, output_size, h, w]
        return torch.sigmoid(out[:, :3]), out[:, 3]


class IndependentPrior(nn.Module):
    def __init__(self, z_size, K):
        super(IndependentPrior, self).__init__()
        self.z_size = z_size
        self.K = K
        self.z_linear = nn.Sequential(
            nn.Linear(self.z_size, 128),
            nn.ELU(True))
        self.z_mu = nn.Linear(128, self.z_size)
        self.z_softplus = nn.Linear(128, self.z_size)

        init_weights(self.z_linear, 'xavier')
        init_weights(self.z_mu, 'xavier')
        init_weights(self.z_softplus, 'xavier')

    def forward(self, slots):
        """
        slots is [N,K,D]
        """
        slots = self.z_linear(slots)  # [N,K,D]
        loc_z = self.z_mu(slots)
        sp_z = self.z_softplus(slots)
        return loc_z, sp_z


class RefinementNetwork(nn.Module):
    """
    EM refinement
    """
    def __init__(self, z_size):
        super(RefinementNetwork, self).__init__()

        self.recurrence = nn.GRU(z_size, z_size)
        self.encoding = nn.Sequential(
            nn.Linear(4 * z_size, 128),
            nn.ELU(True),
            nn.Linear(128, z_size)
        )
        self.loc = nn.Linear(z_size, z_size)
        self.softplus = nn.Linear(z_size, z_size)

        init_weights(self.loc, 'xavier')
        init_weights(self.softplus, 'xavier')
        init_weights(self.encoding, 'xavier')

        self.loc_LN = nn.LayerNorm((z_size,), elementwise_affine=False)
        self.softplus_LN = nn.LayerNorm((z_size,), elementwise_affine=False)

    def forward(self, loss, lamda, hidden_state, eval_mode):
        """
        Args:
            loss: [N] scalar outputs provided to torch.autograd.grad
            lamda: [N*K, 2 * z_size] current posterior parameters
        Returns:
            lamda_next: [N*K, 2 * z_size], the updated posterior parameters
            hidden_state: next recurrent hidden state
        """
        d_lamda = torch.autograd.grad(loss, lamda, create_graph=not eval_mode, \
                                      retain_graph=not eval_mode, only_inputs=True)

        d_loc, d_sp = d_lamda[0].chunk(2, 1)
        d_loc, d_sp = d_loc.contiguous(), d_sp.contiguous()

        d_loc = self.loc_LN(d_loc).detach()
        d_sp = self.softplus_LN(d_sp).detach()

        x = self.encoding(torch.cat([lamda, d_loc, d_sp], 1))
        x = x.unsqueeze(0)
        self.recurrence.flatten_parameters()
        x, hidden_state = self.recurrence(x, hidden_state)
        x = x.squeeze(0)
        return torch.cat([self.loc(x), self.softplus(x)], 1), hidden_state


class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih_1 = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh_1 = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

        self.weight_ih_2 = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh_2 = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        self.zeros = torch.zeros(2 * hidden_size, input_size)

        self.weight_in_1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hn_1 = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.weight_in_2 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hn_2 = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.bias_in = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_hn = nn.Parameter(torch.randn(2 * hidden_size))

        self.zeros_n = torch.zeros(hidden_size, input_size)

    def pad(self, params_1, params_2, zeros):
        params_1 = torch.cat([params_1, zeros], 1)  # [2*hidden,2*input_size]
        params_2 = torch.cat([zeros, params_2], 1)  # [2*hidden,2*input_size]
        params = torch.cat([params_1, params_2], 0)  # [4*hidden,2*input_size]
        return params

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        weight_ih = self.pad(self.weight_ih_1, self.weight_ih_2, self.zeros)
        weight_hh = self.pad(self.weight_hh_1, self.weight_hh_2, self.zeros)

        weight_in = self.pad(self.weight_in_1, self.weight_in_2, self.zeros_n)  # [2*hidden, 2*input_size]
        weight_hn = self.pad(self.weight_hn_1, self.weight_hn_2, self.zeros_n)

        gates = (torch.mm(input, weight_ih.t()) + self.bias_ih + \
                 torch.mm(hx, weight_hh.t()) + self.bias_hh)
        resetgate_1, updategate_1, resetgate_2, updategate_2 = gates.chunk(4, 1)

        resetgate_1 = torch.sigmoid(resetgate_1)
        updategate_1 = torch.sigmoid(updategate_1)
        resetgate_2 = torch.sigmoid(resetgate_2)
        updategate_2 = torch.sigmoid(updategate_2)

        resetgate = torch.cat([resetgate_1, resetgate_2], 1)  # [batch, 2*hidden]
        updategate = torch.cat([updategate_1, updategate_2], 1)

        cellgate = torch.mm(input, weight_in.t()) + self.bias_in + \
                   resetgate * (torch.mm(hx, weight_hn.t()) + self.bias_hn)

        cellgate = torch.tanh(cellgate)  # [batch,2*hidden]

        hy = (1. - updategate) * cellgate + updategate * hx  # [batch,2*hidden]

        return hy, (hy)


class DualGRU(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(DualGRU, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def slot_attention_create_positional_embedding(h, w, batch_size):
    dist_right = torch.linspace(1, 0, w).view(w,1,1).repeat(1,h,1)  # [w,h,1]
    dist_left = torch.linspace(0, 1, w).view(w,1,1).repeat(1,h,1)
    dist_top = torch.linspace(0, 1, h).view(1,h,1).repeat(w,1,1)
    dist_bottom = torch.linspace(1, 0, h).view(1,h,1).repeat(w,1,1)
    return torch.cat([dist_right, dist_left, dist_top, dist_bottom],2).unsqueeze(0).repeat(batch_size,1,1,1)


class HVAENetworks(nn.Module):
    # @net.capture
    def __init__(self, K, z_size, input_size, stochastic_layers, use_DualGRU, batch_size):
        super(HVAENetworks, self).__init__()
        self.K = K
        self.z_size = z_size
        self.batch_size = batch_size
        self.num_stochastic_layers = stochastic_layers
        self.scale = z_size ** -0.5
        self.eps = 1e-8
        self.C, self.H, self.W = input_size
        self.use_DualGRU = use_DualGRU

        h = input_size[1]
        w = input_size[2]
        self.positional_embedding = slot_attention_create_positional_embedding(h, w, batch_size)
        self.pos_embed_projection = nn.Linear(4, 64)
        self.encoder_pt_1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True)
        )
        self.encoder_pt_2 = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.norm_slots = nn.LayerNorm(self.z_size)
        self.norm_mu_pre_ff = nn.LayerNorm(self.z_size)
        self.norm_softplus_pre_ff = nn.LayerNorm(self.z_size)

        self.to_q = nn.Linear(self.z_size, self.z_size, bias=False)
        self.to_k = nn.Linear(64, self.z_size, bias=False)
        self.to_v = nn.Linear(64, self.z_size, bias=False)

        if self.use_DualGRU:
            self.gru = DualGRU(GRUCell, self.z_size, self.z_size)
        else:
            self.gru = nn.GRU(2 * self.z_size, 2 * self.z_size)
            self.project_update = nn.Linear(self.z_size, 2 * self.z_size)
            init_weights(self.project_update, 'xavier')

        self.mlp_mu = nn.Sequential(
            nn.Linear(self.z_size, self.z_size * 2),
            nn.ReLU(True),
            nn.Linear(self.z_size * 2, self.z_size)
        )
        self.mlp_softplus = nn.Sequential(
            nn.Linear(self.z_size, 2 * self.z_size),
            nn.ReLU(True),
            nn.Linear(self.z_size * 2, self.z_size)
        )

        self.image_decoder = ImageDecoder(z_size=z_size, batch_size=batch_size, K=K, input_size=input_size, image_decoder='iodine')
        self.init_posterior = nn.Parameter(torch.cat([torch.zeros(1, self.z_size), torch.ones(1, self.z_size)], 1))
        self.indep_prior = IndependentPrior(z_size=z_size, K=K)

        init_weights(self.encoder_pt_1, 'xavier')
        init_weights(self.encoder_pt_2, 'xavier')
        init_weights(self.to_q, 'xavier')
        init_weights(self.to_k, 'xavier')
        init_weights(self.to_v, 'xavier')
        init_weights(self.mlp_mu, 'xavier')
        init_weights(self.mlp_softplus, 'xavier')
        init_weights(self.image_decoder, 'xavier')

    def forward(self, x, debug):

        pos_embed = self.positional_embedding.to(x.device)
        x = self.encoder_pt_1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [N,H,W,64]
        pos_embed = self.pos_embed_projection(pos_embed)
        x += pos_embed  # [N,H,W,64]
        x = x.view(pos_embed.shape[0], -1, 64)  # [N,64,H*W]
        inputs = self.encoder_pt_2(x)  # [N,H*W,64]

        x_locs, masks, posteriors = [], [], []
        all_samples = {}

        lamda = self.init_posterior.repeat(self.batch_size * self.K, 1)  # [N*K,2*z_size]
        # For L = 0 case...
        loc, sp = lamda.chunk(2, dim=1)
        loc = loc.contiguous()
        sp = sp.contiguous()
        init_posterior = mvn(loc, sp)
        slots = init_posterior.rsample()
        loc_shape = loc.shape
        slots = slots.view(-1, self.K, self.z_size)  # [N, K, D]

        slots_mu = loc
        slots_softplus = sp

        k, v = self.to_k(inputs), self.to_v(inputs)

        for layer in range(self.num_stochastic_layers):
            # scaled dot-product attention
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            q *= self.scale

            dots = torch.einsum('bid,bjd->bij', q, k)
            # dots is [N, K, HW]
            attn = dots.softmax(dim=1) + self.eps
            all_samples[f'attn_{layer}'] = attn.view(self.batch_size, self.K, 1, self.H, self.W).detach()
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            if self.use_DualGRU:
                updates_recurrent = torch.cat([updates, updates], 2)
                slots_recurrent = torch.cat([slots_mu, slots_softplus], 1)
                slots_recurrent, _ = self.gru(
                    updates_recurrent.reshape(1, -1, 2 * self.z_size),
                    slots_recurrent.reshape(-1, 2 * self.z_size))
                slots_mu, slots_softplus = slots_recurrent[0].chunk(2, 1)
            else:
                updates_recurrent = self.project_update(updates)
                slots_recurrent = torch.cat([slots_mu, slots_softplus], 1)
                slots_recurrent, _ = self.gru(
                    updates_recurrent.reshape(1, -1, 2 * self.z_size),
                    slots_recurrent.reshape(1, -1, 2 * self.z_size))
                slots_mu, slots_softplus = slots_recurrent[0].chunk(2, 1)

            slots_mu = slots_mu + self.mlp_mu(self.norm_mu_pre_ff(slots_mu))
            slots_softplus = slots_softplus + self.mlp_softplus(self.norm_softplus_pre_ff(slots_softplus))

            # necessary for autodiff in refinement steps
            lamda = torch.cat([slots_mu, slots_softplus], 1)  # [N*K, 2*z_size]
            slots_mu, slots_softplus = lamda.chunk(2, 1)

            posterior_z = mvn(slots_mu, slots_softplus)
            slots = posterior_z.rsample()

            posteriors += [posterior_z]
            # all_samples[f'posterior_z_{layer}'] = slots.view(-1, self.K, self.z_size)

            if debug:
                # decode
                slots_ = slots.view(-1, self.z_size)
                # [N*K, M, C, H, W], [N*K, M, 1, H, W]
                x_loc, mask_logits = self.image_decoder(slots_)
                slots_ = slots_.view(-1, self.K, self.z_size)
                mask_logits = mask_logits.view(-1, self.K, 1, self.H, self.W)
                mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
                x_loc = x_loc.view(-1, self.K, self.C, self.H, self.W)
                all_samples[f'means_{layer}'] = x_loc
                all_samples[f'masks_{layer}'] = mask_logprobs

            if layer == self.num_stochastic_layers - 1:
                continue

            slots = slots.view(-1, self.K, self.z_size)

        # decode
        slots = slots.view(-1, self.z_size)
        # [N*K, M, C, H, W], [N*K, M, 1, H, W]
        x_loc, mask_logits = self.image_decoder(slots)
        slots = slots.view(-1, self.K, self.z_size)
        mask_logits = mask_logits.view(-1, self.K, 1, self.H, self.W)
        x_loc = x_loc.view(-1, self.K, self.C, self.H, self.W)
        return x_loc, mask_logits, posteriors, all_samples, lamda


class EfficientMORL(nn.Module):
    shortname='effmorl'
    # @net.capture
    def __init__(self, K, z_size, input_size, batch_size, stochastic_layers,
                 log_scale, image_likelihood, geco_warm_start, refinement_iters,
                 bottom_up_prior, reverse_prior_plusplus, use_geco=False, training=None):
        self.training_kwargs = training
        super(EfficientMORL, self).__init__()
        self.K = K
        self.input_size = input_size
        self.stochastic_layers = stochastic_layers
        self.image_likelihood = image_likelihood
        self.batch_size = batch_size
        self.gmm_log_scale = torch.FloatTensor([log_scale])
        self.refinement_iters = refinement_iters
        self.bottom_up_prior = bottom_up_prior
        self.reverse_prior_plusplus = reverse_prior_plusplus
        if self.reverse_prior_plusplus:
            assert not self.bottom_up_prior  # must be false
        self.z_size = z_size

        self.hvae_networks = HVAENetworks(K=K, z_size=z_size, input_size=input_size, stochastic_layers=stochastic_layers, use_DualGRU=True, batch_size=batch_size)
        self.refinenet = RefinementNetwork(z_size=z_size)

        self.h_0 = torch.zeros(1, self.batch_size * self.K, self.z_size)

        self.use_geco = use_geco
        self.geco_warm_start = geco_warm_start
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)
        self.geco = GECO(self.training_kwargs['geco_reconstruction_target'],
                         self.training_kwargs['geco_ema_alpha'])
        self.kl_beta = self.training_kwargs['kl_beta_init']


    def two_stage_inference(self, x, geco, global_step, kl_beta,
                            get_posterior=False, debug=False):
        total_loss = 0.
        final_nll = 0.
        final_kl = 0.
        level_nll = []
        deltas = []
        C, H, W = self.input_size

        all_auxiliary = {}

        x_orig = x.clone()
        x_orig = (x_orig + 1) / 2.  # to (0,1)

        # x_loc are the RGB components
        # mask_logits are the unnormalized masks
        # posteriors is an array of the L Gaussian intermediate posteriors
        # auxiliary_outs are for visualization
        # posterior_lamda are the layer L Gaussian parameters [mu, sigma]
        x_loc, mask_logits, posteriors, auxiliary_outs, posterior_lamda = self.hvae_networks(x, debug)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
        all_auxiliary = {**all_auxiliary, **auxiliary_outs}

        if self.refinement_iters > 0:
            h = self.h_0.to(x.device)

        for refinement_iter in range(self.refinement_iters +1):

            # Recompute x_loc and mask_logits with the updated posterior_lamda
            if refinement_iter > 0:
                # update posterior_lamda
                delta_posterior_lamda, h = self.refinenet(loss, posterior_lamda, h, not self.training)
                posterior_lamda = posterior_lamda + delta_posterior_lamda
                deltas += [torch.mean(torch.norm(delta_posterior_lamda, dim=1)).detach()]
                if refinement_iter == self.refinement_iters:
                    deltas = torch.stack(deltas)

                # decode
                loc, sp = posterior_lamda.chunk(2 ,1)
                posterior = mvn(loc, sp)

                slots = posterior.rsample()

                slots = slots.view(-1, self.z_size)
                # [N*K, M, C, H, W], [N*K, M, 1, H, W]
                x_loc, mask_logits = self.hvae_networks.image_decoder(slots)
                slots = slots.view(-1, self.K, self.z_size)
                mask_logits = mask_logits.view(-1, self.K, 1, H, W)
                x_loc = x_loc.view(-1, self.K, C, H, W)
                mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # image likelihood for computing NLL
            if self.image_likelihood == 'GMM':
                log_var = (2 * self.gmm_log_scale).view(1 ,1 ,1 ,1 ,1).repeat(1, self.K ,1 ,1 ,1).to(x_orig.device)
                nll = gmm_negativeloglikelihood(x_orig, x_loc, log_var, mask_logprobs)
            elif self.image_likelihood == 'Gaussian':
                log_var = (2 * self.gmm_log_scale).view(1 ,1 ,1 ,1).to(x_orig.device)
                nll = gaussian_negativeloglikelihood(x_orig, torch.sum(x_loc * mask_logprobs.exp(), dim=1), log_var)

            # Hierarchical prior computation
            if refinement_iter == 0:
                # top-down kl
                kl_div = torch.zeros(self.batch_size).to(x.device)

                # If using the reversed prior
                if not self.bottom_up_prior:

                    for layer in list(range(self.stochastic_layers))[::-1]:
                        # top layer is standard Gaussian
                        if layer == self.stochastic_layers -1:
                            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                        else:
                            # z^l+1 ~ q(z^{l+1} | z^l, x)
                            z = posteriors[layer +1].rsample()
                            loc_z, sp_z = self.hvae_networks.indep_prior(z.view(-1, self.K, self.z_size))
                            loc_z = loc_z.view(self.batch_size * self.K, -1)
                            sp_z = sp_z.view(self.batch_size * self.K, -1)
                            # p(z^l | z^l+1)
                            prior_z = mvn(loc_z, sp_z)

                        kl = torch.distributions.kl.kl_divergence(posteriors[layer], prior_z)
                        kl = kl.view(self.batch_size, self.K).sum(1)
                        kl_div += kl
                else:
                    for layer in range(self.stochastic_layers):
                        if layer == 0:
                            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                        else:
                            z = posteriors[layer -1].rsample()
                            loc_z, sp_z = self.hvae_networks.indep_prior(z.view(-1, self.K, self.z_size))
                            loc_z = loc_z.view(self.batch_size * self.K, -1)
                            sp_z = sp_z.view(self.batch_size * self.K, -1)
                            prior_z = mvn(loc_z, sp_z)

                        kl = torch.distributions.kl.kl_divergence(posteriors[layer], prior_z)
                        kl = kl.view(self.batch_size, self.K).sum(1)
                        kl_div += kl

            # Refinement step KL
            else:
                if not self.reverse_prior_plusplus or self.stochastic_layers == 0:
                    prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                # else, prior_z = p(z^1 | z^2) when self.reverse_prior_plusplus is True

                # posterior is q(z; \lambda^{(L,i)})
                kl = torch.distributions.kl.kl_divergence(posterior, prior_z)
                kl = kl.view(self.batch_size, self.K).sum(1)
                kl_div = kl

            final_kl = torch.mean(kl_div)
            final_nll = torch.mean(nll)

            all_auxiliary[f'means_{(self.stochastic_layers - 1 +refinement_iter)}'] = x_loc
            all_auxiliary[f'masks_{(self.stochastic_layers - 1 +refinement_iter)}'] = mask_logprobs


            if kl_beta == 0. or self.geco_warm_start > global_step or geco is None:
                loss = torch.mean(nll + kl_beta * kl_div)
            else:
                loss = kl_beta * torch.mean(kl_div) - geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))

            ## N.b. this is the opposite of IODINE; places more weight on the earlier losses than the later ones
            # from refinement
            total_loss += (((self.refinement_iters +2 - refinement_iter +1) / (self.refinement_iters +1)) * loss)

        if get_posterior:
            if self.refinement_iters == 0:
                return posteriors[-1]
            else:
                return posterior
        masks = mask_logprobs.exp()
        all_auxiliary['canvas'] = (x_loc * masks).sum(1)
        all_auxiliary['layers'] = {}
        all_auxiliary['layers']['patch'] = x_loc
        all_auxiliary['layers']['mask'] = masks

        return all_auxiliary, total_loss, final_nll, final_kl, deltas


    def forward(self, x, global_step, debug=False):
        """
        x: [batch_size, C, H, W]
        """
        x = 2*x - 1.
        if self.training:
            for rf in range(len(self.training_kwargs['refinement_curriculum']) - 1, -1, -1):
                if global_step >= self.training_kwargs['refinement_curriculum'][rf][0]:
                    self.refinement_iters = self.training_kwargs['refinement_curriculum'][rf][1]
                    break

        auxiliary_outs, total_loss, nll, kl, deltas = self.two_stage_inference(
            x, self.geco, global_step,
            kl_beta=self.kl_beta, debug=debug)

        outs = {
            'loss': total_loss,
            'rec_loss': nll,
            'kl': kl,
        }
        if len(deltas) > 0:
            outs['deltas'] = deltas


        r = {**outs, **auxiliary_outs}
        return r

    def update_geco(self, global_step, outs):
        if self.training and self.training_kwargs['use_geco']:
            if global_step == self.geco_warm_start:
                self.geco_C_ema = self.geco.init_ema(self.geco_C_ema, outs['rec_loss'])
            elif global_step > self.geco_warm_start:
                self.geco_C_ema = self.geco.update_ema(self.geco_C_ema, outs['rec_loss'])
                self.geco_beta = self.geco.step_beta(self.geco_C_ema,
                                                     self.geco_beta,
                                                     self.training_kwargs['geco_beta_stepsize'])
