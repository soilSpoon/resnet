from enum import Enum
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F


from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


class ResNet(nn.Module):

    def __init__(self, input_size: int, num_layers: List[int], use_bottleneck: bool):
        super().__init__()

        self.input_layer = self.input_block()
        self.output_layer = self.output_block()

        self.hidden_layers = nn.ModuleList([])

        for index, num_layer in enumerate(num_layers):
            self.hidden_layers.append(self.hidden_layer(num_layer, index, use_bottleneck))

    def input_block(self):
        return nn.Sequential(
            nn.Conv2d(input_size, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )


    def output_block(self):
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.Linear(1000, 10),
        ) 

    def hidden_layer(self, residual_block_count: int, layer_index: int, use_bottleneck: bool):
        in_channels = 64 * 2**layer_index
        out_channels = in_channels * (4 if use_bottleneck else 1)

        residual_blocks = nn.ModuleList([])

        for index in range(residual_block_count):
            is_converted = layer_index > 0 and index == 0

            if is_converted:
                residual_block = self.residual_block(
                    3, int(in_channels / 2), out_channels, use_bottleneck, 2
                )
            else:
                residual_block = self.residual_block(
                    3, in_channels, out_channels, use_bottleneck, 1
                )

            residual_blocks.append(residual_block)

        if layer_index > 0:
            convert_block = nn.Conv2d(int(in_channels / 2), in_channels, 1, 2)
        else:
            convert_block = nn.Sequential()

        return nn.ModuleDict(
            {"convert_block": convert_block, "residual_blocks": residual_blocks}
        )

    def residual_block(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        use_bottleneck: bool,
        stride: int = 1,
    ) -> nn.Sequential:
        padding = int(((in_channels - 1) - in_channels + kernel_size) / 2)

        if use_bottleneck:
            residual_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, stride=stride),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(in_channels),
            )
        else:
            residual_block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
            )

        return residual_block

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        last = None

        for layer in self.hidden_layers:
            convert_block = layer["convert_block"]
            residual_blocks = layer["residual_blocks"]

            for index, block in enumerate(residual_blocks):
                before = last
                last = x

                if before is not None:
                    if index == 1:
                        before = convert_block(before)

                    x = torch.add(x, before)

                x = block(x)
                x = F.relu(x)

        x = self.output_block(x)

        return F.log_softmax(x, dim=0)
