import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResNet(nn.Module):
    def __init__(self, input_size: int, num_layers: int):
        super().__init__()

        self.input_block = nn.Conv2d(input_size, 16, 7)

        self.blocks = nn.ModuleList([])

        for _ in range(num_layers):
            self.blocks.append(self.block())

    def block(self):
        return nn.Sequential(
            nn.Conv2d(16, 16, 3, padding="same"),
            nn.BatchNorm2d(16),
        )

    def forward(self, x: Tensor):
        x = self.input_block(x)
        before = x

        for index, block in enumerate(self.blocks):
            x = block(x)

            if index % 2 == 0:
                before = x
            else:
                x += before

            x = F.relu(x)

        return x
