"""
Implementation of IODINE from

"Multi-Object Representation Learning with Iterative Variational Inference"
Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Chris Burgess,
Daniel Zoran, Loic Matthey, Matthew Botvinick, Alexander Lerchner
https://arxiv.org/abs/1903.00450


This (re)-implemetation is draws from re-implementations of
https://github.com/zhixuan-lin/IODINE
https://github.com/MichaelKevinKelly/IODINE/
https://github.com/pemami4911/IODINE.pytorch/
and is based on official code
https://github.com/deepmind/deepmind-research/tree/master/iodine
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from scipy.stats import truncnorm
from torch.nn import init


def truncated_normal_initializer(shape, mean, stddev):
    # compute threshold at 2 std devs
    values = truncnorm.rvs(mean - 2 * stddev, mean + 2 * stddev, size=shape)
    return torch.from_numpy(values).float()


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Modified from: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "truncated_normal":
                m.weight.data = truncated_normal_initializer(
                    m.weight.shape, 0.0, stddev=init_gain
                )
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _softplus_to_std(softplus):
    softplus = torch.min(softplus, torch.ones_like(softplus) * 80)
    return torch.sqrt(torch.log(1.0 + softplus.exp()) + 1e-5)


def normal(loc, pre_softplus_var):
    return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(
            loc, torch.sqrt(F.softplus(pre_softplus_var))
        ),
        1,
    )


def unit_normal(shape, device):
    loc = torch.zeros(shape).to(device)
    scale = torch.ones(shape).to(device)
    return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(loc, scale), 1
    )


def gmm_loglikelihood(x, x_loc, log_var, mask_logprobs):
    """
    mask_logprobs: [N, K, 1, H, W]
    """
    # NLL [batch_size, 1, H, W]
    sq_err = (x.unsqueeze(1) - x_loc).pow(2)
    # log N(x; x_loc, log_var): [N, K, C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    # [N, K, C, H, W]
    log_p_k = mask_logprobs + normal_ll
    # logsumexp over slots [N, C, H, W]
    log_p = torch.logsumexp(log_p_k, dim=1).sum(1, keepdim=True)
    # [batch_size]
    nll = -torch.sum(log_p, dim=[1, 2, 3])

    return nll, {"log_p_k": log_p_k, "normal_ll": normal_ll, "log_p": log_p}


def gaussian_loglikelihood(x_t, x_loc, log_var):
    sq_err = (x_t - x_loc).pow(2)  # [N,C,H,W]
    # log N(x; x_loc, log_var): [N,C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    nll = -torch.sum(normal_ll, dim=[1, 2, 3])  # [N]
    return nll


def rename_state_dict(state_dict, old_strings, new_strings):
    new_state_dict = {}
    for old_string, new_string in zip(old_strings, new_strings):
        for k, v in state_dict.items():
            if old_string in k:
                new_key = k.replace(old_string, new_string)
                new_state_dict[new_key] = v
    for k, v in state_dict.items():
        for old_string in old_strings:
            if old_string in k:
                break
        else:
            new_state_dict[k] = v
    return new_state_dict


def cfg():
    input_size = [3, 64, 64]  # [C, H, W]
    z_size = 64
    K = 4
    inference_iters = 4
    log_scale = math.log(0.10)  # log base e
    refinenet_channels_in = 16
    lstm_dim = 128
    conv_channels = 32
    kl_beta = 1
    geco_warm_start = 1000


class RefinementNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        z_size=64,
        refinenet_channels_in=17,  # should be 17!!!
        conv_channels=64,
        lstm_dim=256,
        k=3,
        stride=2,
    ):
        super(RefinementNetwork, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.conv = nn.Sequential(
            nn.Conv2d(refinenet_channels_in, conv_channels, k, stride, k // 2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, stride, k // 2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, stride, k // 2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, stride, k // 2),
            nn.ELU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(conv_channels, lstm_dim),
            nn.ELU(True),
            # nn.Linear(lstm_dim, lstm_dim),  # Papers says only 1; Official repo has 2
            # nn.ELU(True)
        )

        # self.input_proj = nn.Sequential(
        #     nn.Linear(lstm_dim + 4 * self.z_size, lstm_dim),
        #     nn.ELU(True)
        # )

        # self.lstm = nn.LSTM(lstm_dim, lstm_dim)
        self.lstm = nn.LSTM(lstm_dim + 4 * self.z_size, lstm_dim)
        # self.loc = nn.Linear(lstm_dim, z_size)
        # self.softplus = nn.Linear(lstm_dim, z_size)
        self.ref_head = nn.Linear(lstm_dim, 2 * z_size)

    def forward(self, img_inputs, vec_inputs, h, c):
        """
        img_inputs: [N * K, C, H, W]
        vec_inputs: [N * K, 4*z_size]
        """
        x = self.conv(img_inputs)
        # concat with \lambda and \nabla \lambda
        x = torch.cat([x, vec_inputs], 1)
        # x = self.input_proj(x)
        x = x.unsqueeze(0)  # seq dim
        self.lstm.flatten_parameters()
        out, (h, c) = self.lstm(x, (h, c))
        out = out.squeeze(0)
        # loc = self.loc(out)
        # softplus = self.softplus(out)
        # lamda = torch.cat([loc, softplus], 1)
        lamda = self.ref_head(out)
        return lamda, (h, c)


class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask. This is the architecture used for the
    Multi-dSprites experiment but I haven't seen any issues
    with re-using it for CLEVR. In their paper they slightly
    modify it (e.g., uses 3x3 conv instead of 5x5).
    """

    def __init__(self, input_size, z_size=64, conv_channels=64, k=3):
        super(SpatialBroadcastDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        self.decode = nn.Sequential(
            nn.Conv2d(z_size + 2, conv_channels, k, 1, padding=k // 2 - 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, 1, padding=k // 2 - 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, 1, padding=k // 2 - 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, k, 1, padding=k // 2 - 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, 4, 1, 1),
        )

    @staticmethod
    def spatial_broadcast(z, h, w):
        """
        source: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py
        """
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, z):
        z_sb = SpatialBroadcastDecoder.spatial_broadcast(z, self.h + 8, self.w + 8)
        out = self.decode(z_sb)  # [batch_size * K, output_size, h, w]
        return torch.sigmoid(out[:, :3]), out[:, 3]


class IODINE(nn.Module):
    shortname = "iodine"

    def __init__(
        self,
        batch_size,
        input_size,
        z_size=64,
        K=7,
        inference_iters=5,
        log_scale=math.log(0.10),
        kl_beta=1,
        lstm_dim=256,
        conv_channels=128,
        refinenet_channels_in=17,  # should be 17
    ):
        super(IODINE, self).__init__()

        self.z_size = z_size
        self.input_size = input_size
        self.K = K
        self.inference_iters = inference_iters
        self.batch_size = batch_size
        self.kl_beta = kl_beta
        self.register_buffer(
            "gmm_log_scale", (log_scale * torch.ones(K)).view(1, K, 1, 1, 1)
        )

        self.image_decoder = SpatialBroadcastDecoder(
            input_size, z_size, conv_channels, k=3
        )  # k HERE IS filter size
        self.refine_net = RefinementNetwork(
            input_size, z_size, refinenet_channels_in, conv_channels, lstm_dim
        )

        init_weights(self.image_decoder, "xavier")
        init_weights(self.refine_net, "xavier")

        # learnable initial posterior distribution
        # loc = 0, variance = 1
        self.lamda_0 = nn.Parameter(
            torch.cat([torch.zeros(1, self.z_size), torch.ones(1, self.z_size)], 1)
        )

        # layernorms for iterative inference input
        affine = True  # Paper trains these parameters
        n = self.input_size[1]
        self.layer_norms = torch.nn.ModuleList(
            [
                nn.LayerNorm((1, n, n), elementwise_affine=affine),
                nn.LayerNorm((1, n, n), elementwise_affine=affine),
                nn.LayerNorm((3, n, n), elementwise_affine=affine),
                nn.LayerNorm((1, n, n), elementwise_affine=affine),
                nn.LayerNorm(
                    (self.z_size,), elementwise_affine=affine
                ),  # layer_norm_mean
                nn.LayerNorm(
                    (self.z_size,), elementwise_affine=affine
                ),  # layer_norm_log_scale
            ]
        )

        self.h_0, self.c_0 = (
            torch.zeros(1, self.batch_size * self.K, lstm_dim),
            torch.zeros(1, self.batch_size * self.K, lstm_dim),
        )

    def refinenet_inputs(
        self, image, means, masks, mask_logits, log_p, normal_ll, lamda, loss
    ):
        N, K, C, H, W = image.shape
        # non-gradient inputs
        # 1. image [N, K, C, H, W]
        # 2. means [N, K, C, H, W]
        # 3. masks  [N, K, 1, H, W] (log probs)
        # 4. mask logits [N, K, 1, H, W]
        # 5. mask posterior [N, K, 1, H, W]

        # print(image.shape, means.shape, masks.shape, mask_logits.shape, log_p.shape, normal_ll.shape, lamda.shape, loss.shape)
        mask_ll = normal_ll.sum(dim=2, keepdim=True)
        mask_posterior = mask_ll - torch.logsumexp(
            mask_ll, dim=1, keepdim=True
        )  # logscale
        # 6. pixelwise likelihood [N, K, 1, H, W]
        # log_p_k = torch.logsumexp(log_p_k, dim=1).sum(1)
        log_p_k = log_p.view(-1, 1, 1, H, W).expand(-1, K, -1, -1, -1)
        # 7. LOO likelihood
        # loo_px_l = torch.log(1e-6 + (log_p_k.exp()+1e-6 - (masks + normal_ll.unsqueeze(2).exp())+1e-6)) # [N,K,1,H,W]
        # since counterfactuals have stop grad:
        with torch.no_grad():
            counterfactuals = []
            for i in range(K):
                pll = torch.cat((normal_ll[:, :i], normal_ll[:, i + 1 :]), dim=1)
                msk = torch.cat((mask_logits[:, :i], mask_logits[:, i + 1 :]), dim=1)
                counterfactuals.append(
                    torch.logsumexp(pll + torch.log_softmax(msk, dim=1), dim=1).sum(
                        1, keepdim=True
                    )
                )
            counterfactuals = torch.stack(counterfactuals, dim=1).view(N, K, 1, H, W)
        # 8. Coordinate channel
        x_mesh, y_mesh = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
        )
        # Expand from (h, w) -> (n, k, 1, h, w)
        x_mesh = x_mesh.expand(N, K, 1, -1, -1)
        y_mesh = y_mesh.expand(N, K, 1, -1, -1)

        # 9. \partial L / \partial means
        # [N, K, C, H, W]
        # 10. \partial L/ \partial masks
        # [N, K, 1, H, W]
        # 11. \partial L/ \partial lamda
        # [N*K, 2 * self.z_size]
        d_means, d_masks, d_lamda = torch.autograd.grad(
            loss, [means, masks, lamda], retain_graph=self.training, only_inputs=True
        )

        d_loc_z, d_sp_z = d_lamda.chunk(2, dim=1)
        # d_loc_z, d_sp_z = d_loc_z.contiguous(), d_sp_z.contiguous()

        # Stop gradients and apply LayerNorm
        # dmeans LN + SG
        d_means = self.layer_norms[2](d_means.detach())
        # dmasks LN + SG
        d_masks = self.layer_norms[3](d_masks.detach())
        # log_p LN + SG
        log_p_k = self.layer_norms[0](log_p_k.detach())
        # counterfactual SG + LN
        loo_px_l = self.layer_norms[1](counterfactuals.detach())
        # dzp LN + SG
        d_loc_z = self.layer_norms[4](d_loc_z.detach())
        d_sp_z = self.layer_norms[5](d_sp_z.detach())

        # concat image-size and vector inputs
        image_inputs = torch.cat(
            [
                image,  # 3
                means,  # 3
                masks,  # 1
                mask_logits,  # code seems to provide probs, paper says logits # 1
                mask_posterior,  # in code not in logscale; is here 1
                d_means,  # 3
                d_masks,  # 1
                log_p_k,  # 1
                loo_px_l,  # 1
                x_mesh,  # 1
                y_mesh,  # 1
            ],
            2,
        )
        vec_inputs = torch.cat([lamda, d_loc_z, d_sp_z], 1)

        return image_inputs.view(N * K, -1, H, W), vec_inputs

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes
        and runs inference for T steps
        """
        torch.set_grad_enabled(True)
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]

        # expand lambda_0
        lamda = self.lamda_0.repeat(self.batch_size * self.K, 1)  # [N*K, 2*z_size]
        p_z = unit_normal(
            shape=[self.batch_size * self.K, self.z_size], device=x.device
        )

        total_loss = 0.0
        losses = []
        x_means = []
        masks = []
        h, c = self.h_0, self.c_0
        h = h.to(x.device)
        c = c.to(x.device)

        for i in range(self.inference_iters):
            # sample initial posterior
            loc_z, sp_z = lamda.chunk(2, dim=1)
            # loc_z, sp_z = loc_z.contiguous(), sp_z.contiguous()
            q_z = normal(loc_z, sp_z)
            z = q_z.rsample()

            # Get means and masks
            x_loc, mask_logits = self.image_decoder(z)  # [N*K, C, H, W]
            x_loc = x_loc.view(self.batch_size, self.K, C, H, W)

            # softmax across slots
            mask_logits = mask_logits.view(self.batch_size, self.K, 1, H, W)
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # NLL [batch_size, 1, H, W]
            # log_var = (2 * self.gmm_log_scale)
            nll, ll_outs = gmm_loglikelihood(
                x, x_loc, 2 * self.gmm_log_scale, mask_logprobs
            )

            # KL div
            kl_div = torch.distributions.kl.kl_divergence(q_z, p_z)
            kl_div = kl_div.view(self.batch_size, self.K).sum(1)

            loss = nll + self.kl_beta * kl_div
            loss = torch.mean(loss)

            scaled_loss = ((i + 1.0) / self.inference_iters) * loss
            losses += [scaled_loss]
            total_loss += scaled_loss

            x_means += [x_loc]
            masks += [mask_logprobs]

            # Refinement
            if i == self.inference_iters - 1:
                # after T refinement steps, just output final loss
                continue

            # compute refine inputs
            x_ = x.repeat(self.K, 1, 1, 1).view(self.batch_size, self.K, C, H, W)

            img_inps, vec_inps = self.refinenet_inputs(
                x_,
                x_loc,
                mask_logprobs,
                mask_logits,
                ll_outs["log_p"],
                ll_outs["normal_ll"],
                lamda,
                loss,
            )

            delta, (h, c) = self.refine_net(img_inps, vec_inps, h, c)
            lamda = lamda + delta

        return {
            "canvas": (x_loc * mask_logprobs.exp()).sum(dim=1),
            "loss": total_loss,
            "recon_loss": torch.mean(nll),
            "kl": torch.mean(kl_div),
            "layers": {"patch": x_loc, "mask": mask_logprobs.exp()},
            "z": z,
        }
