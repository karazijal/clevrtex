import warnings

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch.distributions as dist
from torch.distributions.kl import kl_divergence, register_kl
from torch.distributions.utils import logits_to_probs


class OOLBase(nn.Module):
    def __init__(self,
                 n_particles=1,

                 output_dist='normal',
                 output_hparam=0.3,

                 pres_dist_name='bernoulli',
                 z_pres_temperature=1.0,
                 z_pres_prior_p=.01,

                 z_where_prior_loc=[-2.197, -2.197, 0, 0],
                 z_where_prior_scale=[.5, .5, 1, 1],

                 z_what_prior_loc=None,
                 z_what_prior_scale=None,

                 z_depth_prior_loc=None,
                 z_depth_prior_scale=None,

                 z_bg_prior_loc=None,
                 z_bg_prior_scale=None,

                 z_sh_prior_loc=None,
                 z_sh_prior_scale=None
                 ):
        super(OOLBase, self).__init__()

        self.n_particles = n_particles
        self.pres_dist_name = pres_dist_name

        self.output_dist_name = output_dist
        self.output_hparam = output_hparam

        self.z_pres_temperature = z_pres_temperature
        self.z_pres_prior_p = z_pres_prior_p

        self.z_where_prior_loc = z_where_prior_loc
        self.z_where_prior_scale = z_where_prior_scale

        if z_what_prior_loc is not None:
            self.z_what_prior_loc = z_what_prior_loc
        if z_what_prior_scale is not None:
            self.z_what_prior_scale = z_what_prior_scale

        if z_depth_prior_loc is not None:
            self.z_depth_prior_loc = z_depth_prior_loc
        if z_depth_prior_scale is not None:
            self.z_depth_prior_scale = z_depth_prior_scale

        if z_bg_prior_loc is not None:
            self.z_bg_prior_loc = z_bg_prior_loc
        if z_bg_prior_scale is not None:
            self.z_bg_prior_scale = z_bg_prior_scale

        if z_sh_prior_loc is not None:
            self.z_sh_prior_loc = z_sh_prior_loc
        if z_sh_prior_scale is not None:
            self.z_sh_prior_scale = z_sh_prior_scale

        self._prints = set()

    def ensure_correct_tensor(self, maybe_tensor, other=None):
        if not isinstance(maybe_tensor, torch.Tensor):
            if isinstance(maybe_tensor, (int, float)):
                mult = 1
                if other is not None:
                    mult = other.shape[-1]
                maybe_tensor = torch.tensor([maybe_tensor] * mult)
            else:
                maybe_tensor = torch.tensor(maybe_tensor)
        maybe_tensor = maybe_tensor.view(1, -1)
        if other is not None:
            maybe_tensor = maybe_tensor.to(other)
        return maybe_tensor

    def __set_tensor_value(self, maybe_tensor, dest : torch.Tensor):
        if not isinstance(maybe_tensor, torch.Tensor):
            if isinstance(maybe_tensor, (int, float)):
                dest.fill_(maybe_tensor)
                return
            if isinstance(maybe_tensor, (list, tuple, np.ndarray)):
                maybe_tensor = torch.from_numpy(np.array(maybe_tensor)).view(1, -1).to(dest)
        dest.copy_(maybe_tensor, non_blocking=True)

    @property
    def z_pres_temperature(self):
        return self._z_pres_temperature

    @property
    def z_pres_prior_p(self):
        return self._z_pres_prior_p

    @property
    def z_where_prior_loc(self):
        return self._z_where_prior_loc

    @property
    def z_where_prior_scale(self):
        return self._z_where_prior_scale

    @property
    def z_what_prior_loc(self):
        return self._z_what_prior_loc

    @property
    def z_what_prior_scale(self):
        return self._z_what_prior_scale

    @property
    def z_depth_prior_loc(self):
        return self._z_depth_prior_loc

    @property
    def z_depth_prior_scale(self):
        return self._z_depth_prior_scale

    @property
    def z_bg_prior_loc(self):
        return self._z_bg_prior_loc

    @property
    def z_bg_prior_scale(self):
        return self._z_bg_prior_scale

    @property
    def z_sh_prior_loc(self):
        return self._z_sh_prior_loc

    @property
    def z_sh_prior_scale(self):
        return self._z_sh_prior_scale

    @property
    def _tensor_spec(self):
        try:
            p = next(self.parameters())
        except StopIteration:
            p = nn.Parameter(torch.zeros(1))
        return dict(device=p.device, dtype=p.dtype)

    def pres_dist(self, p=None, batch_shape=None, name=None, logits=None):
        # 1. / (1.0 - torch.tensor(1., dtype=torch.float32, device='cuda').clamp(min=eps, max=1. - eps))
        # Seems to give non inf on 1e-7,
        pres_dist_name = name or self.pres_dist_name
        # spec = self._tensor_spec
        # device = spec['device']
        eps = 1e-6
        if pres_dist_name == 'bernoulli':
            d = dist.Bernoulli(p.clamp(min=eps, max=1.0 - eps))
        elif pres_dist_name == 'relaxedbernoulli-hard' or pres_dist_name == 'gumbelsoftmax-st':
            # Gumber-Softmax
            p = p.clamp(min=eps, max=1.0 - eps)
            d = dist.RelaxedBernoulli(self.z_pres_temperature, p)

            def s():
                y = d.rsample()
                y_hard = torch.round(y).to(torch.float)
                return (y_hard - y).detach() + y  # Straight through

            d.sample = s
            if not hasattr(self, 'defined_kl'):
                self.defined_kl = True

                @register_kl(dist.RelaxedBernoulli, dist.RelaxedBernoulli)
                def kl_gumbel_softmax(p, q):
                    n_particles = self.n_particles
                    s = p.rsample()
                    logps = p.log_prob(s) / n_particles
                    logqs = q.log_prob(s) / n_particles
                    for _ in range(1, n_particles):
                        s = p.rsample()
                        logps += p.log_prob(s) / n_particles
                        logqs += q.log_prob(s) / n_particles
                    return logps - logqs

        elif pres_dist_name == 'relaxedbernoulli' or pres_dist_name == 'concrete' or pres_dist_name == 'gumbel-softmax':
            # Gumber-Softmax

            p = p.clamp(min=eps, max=1.0 - eps)
            d = dist.RelaxedBernoulli(self.z_pres_temperature, p)

            def s():
                y = d.rsample()
                if self.training:
                    return y
                y_hard = torch.round(y).to(torch.float)
                return (y_hard - y).detach() + y  # Straight through

            d.sample = s

            if not hasattr(self, 'defined_kl'):
                self.defined_kl = True

                @register_kl(dist.RelaxedBernoulli, dist.RelaxedBernoulli)
                def kl_gumbel_softmax(p, q):
                    n_particles = self.n_particles
                    s = p.rsample()
                    logps = p.log_prob(s) / n_particles
                    logqs = q.log_prob(s) / n_particles
                    for _ in range(1, n_particles):
                        s = p.rsample()
                        logps += p.log_prob(s) / n_particles
                        logqs += q.log_prob(s) / n_particles
                    return logps - logqs

        elif pres_dist_name == 'relaxedbernoulli-bern_kl':
            p = p.clamp(min=eps, max=1.0 - eps)
            d = dist.RelaxedBernoulli(self.z_pres_temperature, p)

            def s():
                y = d.rsample()
                if self.training:
                    return y
                y_hard = torch.round(y).to(torch.float)
                return (y_hard - y).detach() + y  # Straight through -- return y_hard with gradients of y

            d.sample = s

            if not hasattr(self, 'defined_kl'):
                self.defined_kl = True

                @register_kl(dist.RelaxedBernoulli, dist.RelaxedBernoulli)
                def kl_gumbel_softmax(p, q):
                    pb = dist.Bernoulli(p.probs)
                    qb = dist.Bernoulli(q.probs)
                    return kl_divergence(pb, qb)

                @register_kl(dist.RelaxedBernoulli, dist.Bernoulli)
                def kl_relbern_bern(p, q):
                    pb = dist.Bernoulli(p.probs)
                    return kl_divergence(pb, q)

                @register_kl(dist.Bernoulli, dist.RelaxedBernoulli)
                def kl_relbern_bern(p, q):
                    qb = dist.Bernoulli(q.probs)
                    return kl_divergence(p, qb)

        elif pres_dist_name == 'continuousbernoulli':
            #  The continuous Bernoulli: fixing a pervasive error in variational autoencoders,
            #  Loaiza-Ganem G and Cunningham JP, NeurIPS 2019. https://arxiv.org/abs/1907.06845
            p = p.clamp(min=eps, max=1.0 - eps)
            d = dist.ContinuousBernoulli(p)
            d.sample = d.rsample
        else:
            raise ValueError(f"Unknown distribution {pres_dist_name}")
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def where_prior(self, batch_shape=None):
        d = dist.Normal(self.z_where_prior_loc, self.z_where_prior_scale)
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def what_prior(self, batch_shape=None):
        d = dist.Normal(self.z_what_prior_loc, self.z_what_prior_scale)
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def bg_prior(self, batch_shape=None):
        d = dist.Normal(self.z_bg_prior_loc, self.z_bg_prior_scale)
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def depth_prior(self, batch_shape=None):
        d = dist.Normal(self.z_depth_prior_loc, self.z_depth_prior_scale)
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def shape_prior(self, batch_shape=None):
        d = dist.Normal(self.z_sh_prior_loc, self.z_sh_prior_scale)
        if batch_shape:
            d = d.expand(batch_shape)
        return d

    def output_dist(self, canvas):
        eps = 1e-6
        if self.output_dist_name == 'normal':
            d = dist.Normal(canvas, self.output_hparam)
            d.sample = d.rsample
        elif self.output_dist_name == 'bernoulli':
            d = dist.Bernoulli(canvas.clamp(min=eps, max=1. - eps))
        elif self.output_dist_name == 'relaxedbernoulli':
            # Gumber-Softmax
            d = dist.RelaxedBernoulli(torch.tensor([self.output_hparam], device=canvas.device),
                                      canvas.clamp(min=eps, max=1. - eps))
            d.sample = d.rsample
            d.mean = canvas.clamp(min=eps, max=1. - eps)
        elif self.output_dist_name == 'continuousbernoulli':
            #  The continuous Bernoulli: fixing a pervasive error in variational autoencoders,
            #  Loaiza-Ganem G and Cunningham JP, NeurIPS 2019. https://arxiv.org/abs/1907.06845
            d = dist.ContinuousBernoulli(canvas.clamp(min=eps, max=1. - eps))
            d.sample = d.rsample

        elif self.output_dist_name == 'discmixlogistic':
            # warnings.warn("discmixlogistic distrubtion requires logits, which do not support currently support summation")
            if isinstance(self.output_hparam, (tuple, list)):
                nmix, nbits = self.output_hparam
                d = DiscMixLogistic(canvas, nmix, nbits)
            else:
                d = DiscMixLogistic(canvas, num_mix=self.output_hparam)
        elif self.output_dist_name == 'disclogistic':
           d = DiscLogistic(canvas)
        else:
            raise ValueError(f"Unknown output distribution {self.output_dist_name}")
        return d

    @z_pres_temperature.setter
    def z_pres_temperature(self, value):
        if hasattr(self, '_z_pres_temperature'):
            # self._z_pres_temperature = self.ensure_correct_tensor(value, self._z_pres_temperature)
            self.__set_tensor_value(value, self._z_pres_temperature)
        else:
            self.register_buffer('_z_pres_temperature', torch.tensor(value).view(1, -1))

    @z_pres_prior_p.setter
    def z_pres_prior_p(self, value):
        if hasattr(self, '_z_pres_prior_p'):
            # self._z_pres_prior_p = self.ensure_correct_tensor(value, self._z_pres_prior_p)
            self.__set_tensor_value(value, self._z_pres_prior_p)
        else:
            self.register_buffer('_z_pres_prior_p', torch.tensor(value).view(1, -1))

    @z_where_prior_loc.setter
    def z_where_prior_loc(self, value):
        if hasattr(self, '_z_where_prior_loc'):
            # self._z_where_prior_loc = self.ensure_correct_tensor(value, self._z_where_prior_loc)
            self.__set_tensor_value(value, self._z_where_prior_loc)
        else:
            self.register_buffer('_z_where_prior_loc', torch.tensor(value).view(1, -1))

    @z_where_prior_scale.setter
    def z_where_prior_scale(self, value):
        if hasattr(self, '_z_where_prior_scale'):
            # self._z_where_prior_scale = self.ensure_correct_tensor(value, self._z_where_prior_scale)
            self.__set_tensor_value(value, self._z_where_prior_scale)
        else:
            self.register_buffer('_z_where_prior_scale', torch.tensor(value).view(1, -1))

    @z_what_prior_loc.setter
    def z_what_prior_loc(self, value):
        if hasattr(self, '_z_what_prior_loc'):
            # self._z_what_prior_loc = self.ensure_correct_tensor(value, self._z_what_prior_loc)
            self.__set_tensor_value(value, self._z_what_prior_loc)
        else:
            self.register_buffer('_z_what_prior_loc', torch.tensor(value).view(1, -1))

    @z_what_prior_scale.setter
    def z_what_prior_scale(self, value):
        if hasattr(self, '_z_what_prior_scale'):
            # self._z_what_prior_scale = self.ensure_correct_tensor(value, self._z_what_prior_scale)
            self.__set_tensor_value(value, self._z_what_prior_scale)
        else:
            self.register_buffer('_z_what_prior_scale', torch.tensor(value).view(1, -1))

    @z_depth_prior_loc.setter
    def z_depth_prior_loc(self, value):
        if hasattr(self, '_z_depth_prior_loc'):
            # self._z_depth_prior_loc = self.ensure_correct_tensor(value, self._z_depth_prior_loc)
            self.__set_tensor_value(value, self._z_depth_prior_loc)
        else:
            self.register_buffer('_z_depth_prior_loc', torch.tensor(value).view(1, -1))

    @z_depth_prior_scale.setter
    def z_depth_prior_scale(self, value):
        if hasattr(self, '_z_depth_prior_scale'):
            # self._z_depth_prior_scale = self.ensure_correct_tensor(value, self._z_depth_prior_scale)
            self.__set_tensor_value(value, self._z_depth_prior_scale)
        else:
            self.register_buffer('_z_depth_prior_scale', torch.tensor(value).view(1, -1))

    def baseline_parameters(self):
        if hasattr(self, 'baseline') and self.baseline is not None:
            return self.baseline.parameters()
        return ()

    def model_parameters(self):
        if hasattr(self, 'baseline') and self.baseline is not None:
            baseline_parameters = set(self.baseline_parameters())
            for name, param in self.named_parameters():
                if param not in baseline_parameters:
                    yield param
        else:
            return self.parameters()

    @z_bg_prior_loc.setter
    def z_bg_prior_loc(self, value):
        if hasattr(self, '_z_bg_prior_loc'):
            # self._z_bg_prior_loc = self.ensure_correct_tensor(value, self._z_bg_prior_loc)
            self.__set_tensor_value(value, self._z_bg_prior_loc)
        else:
            self.register_buffer('_z_bg_prior_loc', self.ensure_correct_tensor(value))

    @z_bg_prior_scale.setter
    def z_bg_prior_scale(self, value):
        if hasattr(self, '_z_bg_prior_scale'):
            # self._z_bg_prior_scale = self.ensure_correct_tensor(value, self._z_bg_prior_scale)
            self.__set_tensor_value(value, self._z_bg_prior_scale)
        else:
            self.register_buffer('_z_bg_prior_scale', self.ensure_correct_tensor(value))

    @z_sh_prior_loc.setter
    def z_sh_prior_loc(self, value):
        if hasattr(self, '_z_sh_prior_loc'):
            # self._z_sh_prior_loc = self.ensure_correct_tensor(value, self._z_sh_prior_loc)
            self.__set_tensor_value(value, self._z_sh_prior_loc)
        else:
            self.register_buffer('_z_sh_prior_loc', self.ensure_correct_tensor(value))

    @z_sh_prior_scale.setter
    def z_sh_prior_scale(self, value):
        if hasattr(self, '_z_sh_prior_scale'):
            # self._z_sh_prior_scale = self.ensure_correct_tensor(value, self._z_sh_prior_scale)
            self.__set_tensor_value(value, self._z_sh_prior_scale)
        else:
            self.register_buffer('_z_sh_prior_scale', self.ensure_correct_tensor(value))

    def onceprint(self, *args, **kwargs):
        """Just a useful debug function to see shapes when fisrt running"""
        k = '_'.join(str(a) for a in args)
        if k not in self._prints:
            print(*args, **kwargs)
            self._prints.add(k)


class DiscLogistic:
    def __init__(self, param):
        # B, C, H, W = param.size()
        # self.num_c = C // 2
        self.means, self.log_scales = param.chunk(2, 1)
        self.log_scales = self.log_scales.clamp(min=-7.0)

    @property
    def mean(self):
        img = self.means / 2. + 0.5
        return img

    def log_prob(self, samples):
        # assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        # B, C, H, W = samples.size()
        # assert C == self.num_c

        centered = samples - self.means  # B, 3, H, W
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))  # B, 3, H, W

        return log_probs

    def sample(self):
        u = torch.empty(self.means.size(), device=self.means.device).uniform_(1e-5, 1. - 1e-5)  # B, 3, H, W
        x = self.means + torch.exp(self.log_scales) * (torch.log(u) - torch.log(1. - u))  # B, 3, H, W
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x



class DiscMixLogistic:
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]  # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)  # B, 3, 3 * M, H, W
        self.means = l[:, :, :num_mix, :, :]  # B, 3, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)  # B, 3, M, H, W
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])  # B, 3, M, H, W
        self.max_val = 2. ** num_bits - 1

    def one_hot(self, indices, depth, dim):
        indices = indices.unsqueeze(dim)
        size = list(indices.size())
        size[dim] = depth
        y_onehot = torch.zeros(size).to(indices.device)
        y_onehot.scatter_(dim, indices, 1)

        return y_onehot

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'

        # samples = samples.unsqueeze(4)  # B, 3, H , W
        # samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)  # B, 3, M, H, W
        samples = samples.unsqueeze(2).expand(-1, -1, self.num_mix, -1, -1) # B, 3, M, H, W
        mean1 = self.means[:, 0, :, :, :]  # B, M, H, W
        mean2 = self.means[:, 1, :, :, :] + \
                self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]  # B, M, H, W
        mean3 = self.means[:, 2, :, :, :] + \
                self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
                self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]  # B, M, H, W

        # mean1 = mean1.unsqueeze(1)  # B, 1, M, H, W
        # mean2 = mean2.unsqueeze(1)  # B, 1, M, H, W
        # mean3 = mean3.unsqueeze(1)  # B, 1, M, H, W
        means = torch.stack([mean1, mean2, mean3], dim=1)  # B, 3, M, H, W
        centered = samples - means  # B, 3, M, H, W

        inv_stdv = torch.exp(- self.log_scales)

        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)

        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))  # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
        return torch.logsumexp(log_probs, dim=1, keepdim=True)  # B, 1, H, W

    def sample(self, t=1.):
        # Form a (incomplete) sample from relaxed one-hot categorical and then select the highest value.
        # This basically skips a couple of steps of sampling for speed (and memory)
        gumbel = -torch.log(
            - torch.log(torch.empty_like(self.logit_probs).uniform_(1e-5, 1. - 1e-5)))  # B, M, H, W
        sel = self.one_hot(torch.argmax(self.logit_probs + gumbel, 1), self.num_mix, dim=1)  # B, M, H, W
        sel = sel.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)  # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)  # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)  # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.empty_like(means).uniform_(1e-5, 1. - 1e-5)  # B, 3, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))  # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)  # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x

    def rsample(self, t=1.):
        # form a sample from
        gumbel = -torch.log(
            - torch.log(torch.empty_like(self.logit_probs).uniform_(1e-5, 1. - 1e-5)))  # B, M, H, W
        sel = (self.logit_probs + gumbel / t).softmax(dim=1)
        sel = sel.unsqueeze(1).clamp(min=1e-19)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)  # B, 3, H, W
        log_scales = torch.logsumexp(2 * self.log_scales + 2 * sel.log(), dim=2)  * .5  # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)  # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.empty_like(means).uniform_(1e-5, 1. - 1e-5)  # B, 3, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))  # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)  # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x

    @property
    def mean(self):
        probs = self.logit_probs.softmax(dim=1).unsqueeze(1)
        means = torch.sum(self.means * probs, dim=2)
        coeffs = torch.sum(self.coeffs * probs, dim=2)
        x0 = means[:, 0].clamp(-1, 1.)
        x1 = (means[:, 1] + coeffs[:, 0] * x0).clamp(-1, 1.)
        x2 = (means[:, 2] + coeffs[:, 1] + x0 + coeffs[:, 2] * x1).clamp(-1, 1.)
        img = torch.stack([x0, x1, x2]).transpose(0, 1) / 2. + 0.5
        return img

    @staticmethod
    def apply_mask(logits, mask, num_mix=10):
        B, *other, H, W = logits.shape
        lps = logits[..., :num_mix, :, :] + mask.clamp(min=1e-5).log()
        l = logits[..., num_mix:, :, :]
        l = l.view(B, *other[:-1], 3, 3 * num_mix, H, W)

        means = l[..., :, :  num_mix, :, :]
        log_scales = l[..., :, num_mix:2 * num_mix, :, :]
        coeffs = l[..., :, 2 * num_mix:3 * num_mix, :, :]

        means = (means + 1) * mask.unsqueeze(-4) - 1  # push towards -1 rather than 0
        log_scales = log_scales + mask.unsqueeze(-4).clamp(min=1e-5).log()
        coeffs = coeffs * mask.unsqueeze(-4)

        l = torch.cat([means, log_scales, coeffs], -4)
        return torch.cat([lps, l.view(B, *other[:-1], -1, H, W)], -3)

    @staticmethod
    def sum(logits, dim=1, num_mix=10):
        raise NotImplementedError()
        B, *other, H, W = logits.shape
        lps = logits[..., :num_mix, :, :]
        lps = lps.sum(dim=dim)

        l = logits[..., num_mix:, :, :]
        l = l.view(B, *other[:-1], 3, 3 * num_mix, H, W)

        means = l[..., :, :  num_mix, :, :]
        log_scales = l[..., :, num_mix:2 * num_mix, :, :]
        coeffs = l[..., :, 2 * num_mix:3 * num_mix, :, :]

        means = means.sum(dim=dim)
        log_scales = torch.logsumexp(2 * log_scales, dim=dim)  * .5
        # This is how much channels add... cannot be a sum, but a mean is only a guess
        # This this in after tanh
        # tanh_coeffs = coeffs.tanh().mean(dim=dim).clamp(-1.+1e-5, 1.-1e-5)
        # coeffs = .5 * ((1+tanh_coeffs).log() - (1-tanh_coeffs).log())
        coeffs = coeffs.sum(dim=dim)
        other = list(other)
        del other[dim]
        l = torch.cat([means, log_scales, coeffs], -4)
        return torch.cat([lps, l.view(B, *other[:-1], -1, H, W)], -3)

