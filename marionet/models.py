"""Model architecture.

Defintions:
    - layer: refers to a raster layer in the composite. Each layer is assembled
      from multiple patches. A rendering can have multiple layers ordered from
      back to front.
"""
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn

from .partialconv import PartialConv2d


class Dictionary(nn.Module):
    def __init__(
        self, num_classes, patch_size, n_ch, bottleneck_size=128, no_layernorm=False
    ):
        super().__init__()

        self.patch_size = patch_size
        self.n_ch = n_ch
        self.no_layernorm = no_layernorm

        self.latent = nn.Parameter(th.randn(num_classes, bottleneck_size))
        self.decode = nn.Sequential(
            nn.Linear(bottleneck_size, 8 * bottleneck_size),
            nn.GroupNorm(8, 8 * bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(8 * bottleneck_size, n_ch * patch_size[0] * patch_size[1]),
            nn.Sigmoid(),
        )

    def forward(self, x=None):
        if x is None and not self.no_layernorm:
            x = F.layer_norm(self.latent, (self.latent.shape[-1],))
        out = self.decode(x).view(-1, self.n_ch, *self.patch_size)
        return out, x


class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, return_mask=True):
        super().__init__()
        self.return_mask = return_mask
        self.conv1 = PartialConv2d(in_ch, out_ch, 3, padding=1, return_mask=True)
        self.conv2 = PartialConv2d(
            out_ch, out_ch, 3, padding=1, stride=2, return_mask=True
        )
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.nonlinearity = nn.LeakyReLU(inplace=True)

    def forward(self, x, mask=None):
        y, mask = self.conv1(x, mask)
        y = self.nonlinearity(self.norm1(y))
        y, mask = self.conv2(y, mask)
        y = self.nonlinearity(self.norm2(y))

        if self.return_mask:
            return y, mask
        else:
            return y


class Encoder(nn.Module):
    """Encodes image data into a grid of patch latent codes used for affinity
    matching.
    The encoder is a chain of blocks that each dowsamples by 2x. It outputs
    a list of latent codes for the patches in each layer. Each layer can have a
    variable (power of two) number of patches. The latent codes are predicted
    from the corresponding block.
    im
     |
     V
    block1 -> (optional) latent codes for all layers with 2x2 patches
     |
     V
    block2 -> (optional) latent codes for all layers with 4x4 patches
     |
     V
    ...    -> ...
    Args:
        num_channels(int): number of image channels (e.g. 4 for RGBA images).
        canvas_size(int): size of the (square) input image.
        layer_sizes(list of int): list of patch count along the x (resp. y)
            dimension for each layer. The number of layers is the length of
            the list.
    """

    def __init__(
        self, num_channels, canvas_size, layer_sizes, dim_z=1024, no_layernorm=False
    ):
        super().__init__()

        self.canvas_size = canvas_size
        self.layer_sizes = layer_sizes
        self.num_channels = num_channels
        self.no_layernorm = no_layernorm

        num_ds = int(np.log2(canvas_size / min(layer_sizes)))

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(num_ds):
            in_ch = num_channels if i == 0 else dim_z
            self.blocks.append(_DownBlock(in_ch, dim_z, return_mask=True))

            for lsize in layer_sizes:
                if canvas_size // (2 ** (i + 1)) == lsize:
                    self.heads.append(PartialConv2d(dim_z, dim_z, 3, padding=1))

    def forward(self, x):
        out = [None] * len(self.layer_sizes)

        y = x
        mask = None
        for _block in self.blocks:
            y, mask = _block(y, mask)  # encoding + downsampling step

            # Look for layers whose spatial dimension match the current block
            for i, l in enumerate(self.layer_sizes):
                if y.shape[-1] == l:
                    # size match, output the latent codes for this layer
                    out[i] = self.heads[i](y, mask).permute(0, 2, 3, 1).contiguous()
                    if not self.no_layernorm:
                        out[i] = F.layer_norm(out[i], (out[i].shape[-1],))

        # Check all outputs were set
        for o in out:
            if o is None:
                raise RuntimeError("Unexpected output count for Encoder.")

        return out


class Model(nn.Module):
    def __init__(
        self,
        learned_dict,
        layer_size,
        num_layers,
        patch_size=1,
        canvas_size=128,
        dim_z=128,
        shuffle_all=False,
        bg_color=None,
        bg_size=(128, 512),
        num_bg=1,
        dim_bg=4,
        no_layernorm=False,
        no_spatial_transformer=False,
        spatial_transformer_bg=False,
        straight_through_probs=False,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // layer_size
        self.dim_z = dim_z
        self.shuffle_all = shuffle_all
        self.no_spatial_transformer = no_spatial_transformer
        self.spatial_transformer_bg = spatial_transformer_bg
        self.straight_through_probs = straight_through_probs

        self.im_encoder = Encoder(
            3, canvas_size, [layer_size] * num_layers, dim_z, no_layernorm=no_layernorm
        )

        self.project = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.LayerNorm(dim_z, elementwise_affine=False)
            if not no_layernorm
            else nn.Identity(),
        )

        self.encoder_xform = Encoder(
            7, self.patch_size * 2, [1], dim_z, no_layernorm=no_layernorm
        )
        self.probs = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GroupNorm(8, dim_z),
            nn.LeakyReLU(),
            nn.Linear(dim_z, 1),
            nn.Sigmoid(),
        )

        if self.no_spatial_transformer:
            self.xforms_x = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size + 1),
                nn.Softmax(dim=-1),
            )
            self.xforms_y = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size + 1),
                nn.Softmax(dim=-1),
            )
        else:
            self.shifts = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh(),
            )

        self.learned_dict = learned_dict

        if bg_color is None:
            self.bg_encoder = Encoder(
                3, canvas_size, [1], dim_bg, no_layernorm=no_layernorm
            )
            if self.spatial_transformer_bg:
                self.bg_shift = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 1),
                    nn.Tanh(),
                )
            else:
                self.bg_project = nn.Sequential(
                    nn.Linear(dim_bg, dim_bg),
                    nn.LayerNorm(dim_bg, elementwise_affine=False)
                    if not no_layernorm
                    else nn.Identity(),
                )

                # self.bg_x = nn.Sequential(
                #     nn.Linear(dim_z, dim_z),
                #     nn.GroupNorm(8, dim_z),
                #     nn.LeakyReLU(),
                #     nn.Linear(dim_z, bg_size[-1] - self.canvas_size + 1),
                #     nn.Softmax(dim=-1)
                # )
        else:
            self.bg_color = nn.Parameter(th.tensor(bg_color), requires_grad=False)

    def forward(self, im, bg, hard=False, custom_dict=None, rng=None, custom_bg=None):
        bs = im.shape[0]

        learned_dict, dict_codes = self.learned_dict()
        if rng is not None:
            learned_dict = learned_dict[rng]
            dict_codes = dict_codes[rng]

        im_codes = th.stack(self.im_encoder(im), dim=1)
        probs = self.probs(
            im_codes.flatten(0, 3)
        )  # flatten to handle the groupnorm (which expects channels to dim=1)
        probs_layer_grid = probs.view(*im_codes.shape[:-1], -1)
        if self.straight_through_probs:
            probs = probs.round() - probs.detach() + probs
        proj_im_codes = self.project(im_codes)

        logits = (proj_im_codes @ dict_codes.transpose(0, 1)) / np.sqrt(
            im_codes.shape[-1]
        )
        weights = F.softmax(logits, dim=-1)
        # B, Nl, Gh, Gw, Nc
        patches = (weights[..., None, None, None] * learned_dict).sum(4)
        patches = patches.flatten(0, 3)

        im_patches = F.pad(im, (self.patch_size // 2,) * 4)
        im_patches = im_patches.unfold(2, self.patch_size * 2, self.patch_size).unfold(
            3, self.patch_size * 2, self.patch_size
        )
        im_patches = (
            im_patches.reshape(
                bs,
                3,
                self.layer_size,
                self.layer_size,
                2 * self.patch_size,
                2 * self.patch_size,
            )
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        # B Gh Gw C pH pW

        im_patches = (
            im_patches[:, None].repeat(1, self.num_layers, 1, 1, 1, 1, 1).flatten(0, 3)
        )
        # (B Nl Gh Gw) C pH pW

        codes_xform = (
            self.encoder_xform(th.cat([im_patches, patches], dim=1))[0]
            .squeeze(-2)
            .squeeze(-2)
        )
        if hard:
            weights = th.eye(weights.shape[-1]).to(weights)[weights.argmax(-1)]
            probs = probs.round()
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        if custom_dict is not None:
            learned_dict = custom_dict
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        patches = patches * probs[:, :, None, None]

        if self.no_spatial_transformer:
            xforms_x = self.xforms_x(codes_xform)
            xforms_y = self.xforms_y(codes_xform)

            if hard:
                xforms_x = th.eye(xforms_x.shape[-1]).to(xforms_x)[xforms_x.argmax(-1)]
                xforms_y = th.eye(xforms_y.shape[-1]).to(xforms_y)[xforms_y.argmax(-1)]

            patches = F.pad(patches, (self.patch_size // 2,) * 4)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_y[:, None, :, None, None]).sum(2)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_x[:, None, :, None, None]).sum(2)
        else:
            shifts = self.shifts(codes_xform) / 2
            theta = th.eye(2)[None].repeat(shifts.shape[0], 1, 1).to(shifts)
            theta = th.cat([theta, -shifts[:, :, None]], dim=-1)
            grid = F.affine_grid(
                theta,
                [patches.shape[0], 1, self.patch_size * 2, self.patch_size * 2],
                align_corners=False,
            )

            patches_rgb, patches_a = th.split(patches, [3, 1], dim=1)
            patches_rgb = F.grid_sample(
                patches_rgb,
                grid,
                align_corners=False,
                padding_mode="border",
                mode="bilinear",
            )
            patches_a = F.grid_sample(
                patches_a,
                grid,
                align_corners=False,
                padding_mode="zeros",
                mode="bilinear",
            )
            patches = th.cat([patches_rgb, patches_a], dim=1)

        patches = patches.view(
            bs,
            self.num_layers,
            self.layer_size,
            self.layer_size,
            -1,
            2 * self.patch_size,
            2 * self.patch_size,
        ).permute(0, 1, 4, 2, 5, 3, 6)
        # print(f'patches {patches.shape}, probs {probs.shape} {probs_layer_grid.shape} {probs_layer_grid.round().flatten(1).sum(1)}')

        group1 = patches[..., ::2, :, ::2, :].contiguous()
        s = self.patch_size * 2
        group1_sem = []
        for i in range(group1.shape[-4]):
            for j in range(group1.shape[-2]):
                p = th.zeros(
                    (
                        bs,
                        self.num_layers,
                        patches.shape[2],
                        self.canvas_size,
                        self.canvas_size,
                    ),
                    dtype=patches.dtype,
                    device=patches.device,
                    requires_grad=False,
                )
                p[..., :, i * s : (i + 1) * s, j * s : (j + 1) * s] = group1[
                    ..., i, :, j, :
                ]
                group1_sem.append(p)
        group1_sem = th.stack(group1_sem, dim=2)

        group1 = group1.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group1 = group1[..., self.patch_size // 2 :, self.patch_size // 2 :]
        group1_sem = group1_sem[..., self.patch_size // 2 :, self.patch_size // 2 :]
        group1 = F.pad(group1, (0, self.patch_size // 2, 0, self.patch_size // 2))
        group1_sem = F.pad(
            group1_sem, (0, self.patch_size // 2, 0, self.patch_size // 2)
        )

        group2 = patches[..., 1::2, :, 1::2, :].contiguous()

        group2_sem = []
        for i in range(group2.shape[-4]):
            for j in range(group2.shape[-2]):
                p = th.zeros(
                    (
                        bs,
                        self.num_layers,
                        patches.shape[2],
                        self.canvas_size,
                        self.canvas_size,
                    ),
                    dtype=patches.dtype,
                    device=patches.device,
                    requires_grad=False,
                )
                p[..., :, i * s : (i + 1) * s, j * s : (j + 1) * s] = group2[
                    ..., i, :, j, :
                ]
                group2_sem.append(p)
        group2_sem = th.stack(group2_sem, dim=2)

        group2 = group2.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group2 = group2[..., : -self.patch_size // 2, : -self.patch_size // 2]
        group2_sem = group2_sem[..., : -self.patch_size // 2, : -self.patch_size // 2]

        group2 = F.pad(group2, (self.patch_size // 2, 0, self.patch_size // 2, 0))
        group2_sem = F.pad(
            group2_sem, (self.patch_size // 2, 0, self.patch_size // 2, 0)
        )

        group3 = patches[..., 1::2, :, ::2, :].contiguous()

        group3_sem = []
        for i in range(group3.shape[-4]):
            for j in range(group3.shape[-2]):
                p = th.zeros(
                    (
                        bs,
                        self.num_layers,
                        patches.shape[2],
                        self.canvas_size,
                        self.canvas_size,
                    ),
                    dtype=patches.dtype,
                    device=patches.device,
                    requires_grad=False,
                )
                p[..., :, i * s : (i + 1) * s, j * s : (j + 1) * s] = group3[
                    ..., i, :, j, :
                ]
                group3_sem.append(p)
        group3_sem = th.stack(group3_sem, dim=2)

        group3 = group3.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group3 = group3[..., : -self.patch_size // 2, self.patch_size // 2 :]
        group3_sem = group3_sem[..., : -self.patch_size // 2, self.patch_size // 2 :]
        group3 = F.pad(group3, (0, self.patch_size // 2, self.patch_size // 2, 0))
        group3_sem = F.pad(
            group3_sem, (0, self.patch_size // 2, self.patch_size // 2, 0)
        )

        group4 = patches[..., ::2, :, 1::2, :].contiguous()

        group4_sem = []
        for i in range(group4.shape[-4]):
            for j in range(group4.shape[-2]):
                p = th.zeros(
                    (
                        bs,
                        self.num_layers,
                        patches.shape[2],
                        self.canvas_size,
                        self.canvas_size,
                    ),
                    dtype=patches.dtype,
                    device=patches.device,
                    requires_grad=False,
                )
                p[..., :, i * s : (i + 1) * s, j * s : (j + 1) * s] = group4[
                    ..., i, :, j, :
                ]
                group4_sem.append(p)
        group4_sem = th.stack(group4_sem, dim=2)

        group4 = group4.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group4 = group4[..., self.patch_size // 2 :, : -self.patch_size // 2]
        group4_sem = group4_sem[..., self.patch_size // 2 :, : -self.patch_size // 2]

        group4 = F.pad(group4, (self.patch_size // 2, 0, 0, self.patch_size // 2))
        group4_sem = F.pad(
            group4_sem, (self.patch_size // 2, 0, 0, self.patch_size // 2)
        )

        layers = th.stack([group1, group2, group3, group4], dim=2)
        layers_sem = th.stack([group1_sem, group2_sem, group3_sem, group4_sem], dim=2)
        layers_out = layers.clone()
        # print(f'{layers_out.shape} {probs_layer_grid.shape} {probs_layer_grid.round().flatten(1).sum(1)}')

        if self.shuffle_all:
            layers = layers.flatten(1, 2)[:, th.randperm(4 * self.num_layers)]
        else:
            perm = th.randperm(4)
            layers = layers[:, :, perm].flatten(1, 2)
            layers_sem = layers_sem[:, :, perm].flatten(1, 3)

        # print(f'{layers.shape}, {probs.shape}, layers_sem {layers_sem.shape}')
        if bg is not None:
            bg_codes = self.bg_encoder(im)
            if not self.spatial_transformer_bg:
                bg_codes = torch.stack(bg_codes, dim=1).squeeze(-2).squeeze(-2)
                bg_dict, bg_dict_codes = bg
                bg = bg_dict

                # print(bg_dict.shape, bg_dict_codes.shape, bg_codes.shape)
                proj_bg_codes = self.bg_project(bg_codes)
                # print(proj_bg_codes.shape)
                bg_logits = (proj_bg_codes @ bg_dict_codes.transpose(0, 1)) / np.sqrt(
                    bg_codes.shape[-1]
                )
                bg_weights = F.softmax(bg_logits, dim=-1)
                # print(bg_weights.shape)
                out = (
                    (bg_weights[..., None, None, None] * bg_dict[None, None])
                    .sum(-4)
                    .squeeze(1)
                )
                # print(bgs.shape)
                # bg_x = self.bg_x(bg_codes)
                # bgs = bg.unfold(-1, self.canvas_size, 1)
                # print(bgs.shape, bg_x.shape)
                # out = (bgs * bg_x[:, None, None, :, None]).sum(3)
            else:
                bg_codes = bg_codes[0].squeeze(-2).squeeze(-2)
                shift = self.bg_shift(bg_codes) * 3 / 4
                shift = th.cat([shift, th.zeros_like(shift)], dim=-1)
                theta = th.eye(2)[None].repeat(shift.shape[0], 1, 1).to(shift)
                theta[:, 0, 0] = 1 / 4
                theta = th.cat([theta, -shift[:, :, None]], dim=-1)
                grid = F.affine_grid(
                    theta,
                    [bs, 1, self.canvas_size, self.canvas_size],
                    align_corners=False,
                )

                out = F.grid_sample(
                    bg.repeat(bs, 1, 1, 1),
                    grid,
                    align_corners=False,
                    padding_mode="border",
                    mode="bilinear",
                )

        else:
            if custom_bg is not None:
                out = (
                    custom_bg[None, :, None, None]
                    .clamp(0, 1)
                    .repeat(bs, 1, self.canvas_size, self.canvas_size)
                )
            else:
                out = (
                    self.bg_color[None, :, None, None]
                    .clamp(0, 1)
                    .repeat(bs, 1, self.canvas_size, self.canvas_size)
                )
            bg = (
                self.bg_color[None, :, None, None]
                .clamp(0, 1)
                .repeat(1, 1, self.canvas_size, self.canvas_size)
            )

        bg_out = out
        rgb, a = th.split(layers, [3, 1], dim=2)
        rgb_sem, a_sem = th.split(layers_sem, [3, 1], dim=2)

        for i in range(4 * self.num_layers):
            out = (1 - a[:, i]) * out + a[:, i] * rgb[:, i]

        pats = torch.zeros_like(rgb_sem)
        masks = torch.zeros_like(a_sem)
        occl = out.new_ones((bs, 1, self.canvas_size, self.canvas_size))
        # print(occl.shape, a_sem.shape, rgb_sem.shape)
        for i in reversed(list(range(rgb_sem.shape[1]))):
            pats[:, i] = rgb_sem[:, i] * a_sem[:, i] * occl
            masks[:, i] = a_sem[:, i] * occl
            occl = occl * (1 - a_sem[:, i])

        ret = {
            "weights": weights,
            "probs": probs.view(bs, self.num_layers, -1),
            "layers": layers_out,
            "patches": patches,
            "l": {"patch": pats, "mask": masks},
            "dict_codes": dict_codes,
            "im_codes": im_codes.flatten(0, 1),
            "reconstruction": out,
            "dict": learned_dict,
            "background": bg,
            "bg": bg_out,
        }

        if not self.no_spatial_transformer:
            ret["shifts"] = shifts

        return ret
