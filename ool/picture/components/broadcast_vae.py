import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Sequential):
    def __init__(self, inc, z_dim):
        super(ConvEncoder, self).__init__(
            nn.Conv2d(inc, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_dim)
        )

    def forward(self, x):
        x = super(ConvEncoder, self).forward(x)
        mu, std = x.chunk(2, 1)
        return mu, F.softplus(std)


class BroadcastDecoder(nn.Sequential):
    def __init__(self, z_dim, out_ch):
        super(BroadcastDecoder, self).__init__(
            nn.Conv2d(z_dim + 2, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0),
        )
        self.out_shape = out_ch

    def forward(self, x, out_shape, sigmoid=True):
        h, w = out_shape[-2:]
        h += 8
        w += 8
        x = x.view(x.size(0), -1, 1, 1).repeat(1, 1, h, w)
        hs = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        ws = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        c = torch.stack(torch.meshgrid(hs, ws)).view(1, 2, h, w).repeat(
            x.size(0), 1, 1, 1)
        x = torch.cat([x, c], dim=1)
        x = super(BroadcastDecoder, self).forward(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
