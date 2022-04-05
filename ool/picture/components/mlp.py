import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_f, out_f, h=[], passf=0):
        super(MLP, self).__init__()
        prev_out = in_f
        ofs = h + [out_f + passf]
        self.layers = nn.ModuleList()
        for of in ofs:
            self.layers.append(nn.Linear(in_features=prev_out, out_features=of))
            prev_out = of
        self.out_f = out_f

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers[:-1]:
            x = torch.relu(l(x))
        x = self.layers[-1](x)
        return x[:, : self.out_f], torch.relu(x[:, self.out_f :])
