import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResNet(nn.Module):
    RESIDUAL_BLOCK_OPTIONS = {
        34: [(3, 64, 64, 3), (3, 128, 128, 4), (3, 256, 256, 6), (3, 512, 512, 3)]
    }

    def __init__(self, input_size: int, num_layers: int):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(input_size, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.hidden_blocks = nn.ModuleList([])

        for _ in range(3):
            self.hidden_blocks.append(self.residual_block(3, 64, 64))

        self.convert_blocks = nn.ModuleDict({"4": nn.Conv2d(64, 128, 1, 2)})

        for num in range(4):
            is_first = num == 0
            in_channels = 64 if is_first else 128
            stride = 2 if is_first else 1

            self.hidden_blocks.append(self.residual_block(3, in_channels, 128, stride))

    def residual_block(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> nn.Sequential:
        padding = int(((in_channels - 1) - in_channels + kernel_size) / 2)

        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, stride=stride
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        before = None

        for index, block in enumerate(self.hidden_blocks):
            tmp = before
            before = x

            if tmp is not None:
                key = str(index)
                if key in self.convert_blocks:
                    tmp = self.convert_blocks[key](tmp)

                x += tmp

            x = block(x)
            x = F.relu(x)

        return x
