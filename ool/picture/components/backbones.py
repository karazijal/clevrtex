import torch
import torch.nn as nn

class SPACEBackbone(nn.Sequential):
    """Extra down = 0 for the original"""
    def __init__(self, inc, extra_down=0):
        layers = [
            nn.Conv2d(inc, 16, 3, 2, 1),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 3, 2, 1),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 3, 2, 1),  # This has been changed from 4x4 kernel to 3x3 kernel
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 256, 3, 1 if extra_down==0 else 2, 1),  # This has been changed from stride 1 to stride 2
            nn.CELU(),
            nn.GroupNorm(32, 256),
        ]
        for _ in range(1, extra_down):
            layers.extend([
                nn.Conv2d(256, 256, 3, 2, 1),
                nn.CELU(),
                nn.GroupNorm(32, 256),
            ])
        layers.extend([
            nn.Conv2d(256, 128, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128)
        ])
        super(SPACEBackbone, self).__init__(*layers)
