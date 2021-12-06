"""
Implementation adjusted from https://github.com/lucidrains/slot-attention
"""


import torch
import torch.nn as nn
import torch.nn.functional as f


class SlotAttention(nn.Module):
    shortname = 'slot'
    def __init__(self, dim, num_iterations, num_slots, mlp_hidden_size, eps=1e-8):
        super(SlotAttention, self).__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.eps = eps

        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        mlp_hidden_size = max(dim, mlp_hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, input):
        B, N, C = input.shape
        device = input.device
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        input = self.norm_input(input)
        k, v = self.to_k(input), self.to_v(input)

        for _ in range(self.num_iterations):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bsc,bnc->bsn', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bnc,bsn->bsc', v, attn)

            slots = self.gru(
                updates.reshape(-1, C),
                slots_prev.reshape(-1, C)
            )

            slots = slots.reshape(B, -1, C)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots



class SlotAttentionAutoEncoder(nn.Module):
    shortname = 'slota'
    def __init__(self, input_shape, num_slots, num_iterations):
        super(SlotAttentionAutoEncoder, self).__init__()
        ic, *resolution = input_shape
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(ic, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.decoder_initial_size = (8,8)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, ic+1, kernel_size=3, stride=1, padding=1),
        )

        self.encoder_pos = SoftPositionEmbed(64, self.resolution)
        self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm([64, *resolution])  # The original seems to normalise accross channels!
        self.mlp = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1)
        )

        self.slot_attention = SlotAttention(64, num_iterations=self.num_iterations, num_slots=self.num_slots, mlp_hidden_size=128)

    def forward(self, img):
        x = self.encoder_cnn(img)
        x = self.encoder_pos(x)
        x = self.mlp(self.layer_norm(x))
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2]*x.shape[3], x.shape[1])
        # Now in [B, HxW, C] shape
        # print(img.shape, x.shape)

        slots = self.slot_attention(x)
        # print(slots.shape)
        # Now in [B, N, 64] shape
        x = slots[:, :, :, None, None].expand(-1, -1, -1, *self.decoder_initial_size).view(-1, slots.shape[-1], *self.decoder_initial_size)
        # Now in [B*N, 64, h, w]
        x = self.decoder_pos(x)
        # print(x.shape)
        x = self.decoder_cnn(x)
        if x.shape[-2:] != img.shape[-2:]:
            x = f.interpolate(x, size=img.shape[-2:], mode='bicubic', align_corners=False)
        # Now in [B*N, 4, H, W]

        x = x.reshape(img.shape[0], self.num_slots, *x.shape[1:])

        # Now in [B, N, 4, H, W]
        # print(x.shape)
        recons = x[:, :, :-1]
        masks = x[:, :, -1:]
        masks = masks.softmax(axis=1)  # Over slot axis

        recon_combined = (recons * masks).sum(axis=1)
        r = {
            'canvas': recon_combined,
            'layers': {
                'mask': masks,
                'patch': recons,
            }
        }
        if self.training:
            r['loss'] = f.mse_loss(recon_combined, img, reduction='none').sum((1, 2, 3)).mean()

        return r

class SoftPositionEmbed(nn.Module):
    def __init__(self, hsize, resolution):
        super(SoftPositionEmbed, self).__init__()

        h, w = resolution[-2:]
        hs = torch.linspace(0., 1., h)
        ws = torch.linspace(0., 1., w)
        c = torch.stack(torch.meshgrid(hs, ws), dim=0)
        grid = torch.cat([c, 1-c], dim=0)[None]

        self.register_buffer('grid', grid)
        self.aff = nn.Conv2d(4, hsize, 1, bias=True)

    def forward(self, input):
        return input + self.aff(self.grid)
