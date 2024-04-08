from torch import Tensor, nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        out: Tensor = self.squeeze(x)
        out = out.view(b, c)
        out = self.excitation(out)
        out = out.view(b, c, 1, 1)
        return x * out.expand_as(x)
