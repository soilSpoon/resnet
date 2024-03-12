from enum import Enum
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F


from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


def conv(
    kernel_size: int, in_channels: int, out_channels: int, stride: int = 1
) -> nn.Module:
    padding = (kernel_size - 1) / 2 * stride

    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(out_channels),
    )


def conv7x7(in_channels: int, out_channels: int, stride: int = 1):
    return conv(7, in_channels, out_channels, stride=stride)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    return conv(1, in_channels, out_channels, stride=stride)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return conv(3, in_channels, out_channels, stride=stride)


def normal_residual_block(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=stride),
        nn.ReLU(),
        conv3x3(out_channels, out_channels),
    )


def bottleneck_residual_block(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Sequential(
        conv1x1(in_channels, in_channels, stride=stride),
        nn.ReLU(),
        conv3x3(in_channels, in_channels),
        nn.ReLU(),
        conv1x1(in_channels, out_channels),
    )


class ResNet(nn.Module):

    def __init__(self, input_size: int, num_layers: List[int], use_bottleneck: bool):
        super().__init__()

        self.use_bottleneck = use_bottleneck

        self.input_layer = self.input_block(input_size)
        self.output_layer = self.output_block()
        self.hidden_layers = nn.ModuleList([])

        for layer_index, num_layer in enumerate(num_layers):
            self.hidden_layers.append(
                self.hidden_layer(num_layer, layer_index, use_bottleneck)
            )

    def input_block(self, input_size):
        return nn.Sequential(
            conv7x7(input_size, 64, stride=2, padding=3),
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

    def hidden_layer(self, residual_block_count: int, layer_index: int):
        is_not_first_layer = layer_index != 0

        in_channels = 64 * 2**layer_index
        out_channels_weight = 4 if self.use_bottleneck else 1
        out_channels = in_channels * out_channels_weight

        residual_blocks = nn.ModuleList([])

        for block_index in range(residual_block_count):
            is_first_block = block_index == 0
            is_channel_changed = is_not_first_layer and is_first_block
            in_channels_weight = 0.5 if is_channel_changed else 1

            residual_block = self.residual_block(
                int(in_channels * in_channels_weight), out_channels, 2
            )

            residual_blocks.append(residual_block)

        convert_block = (
            nn.Conv2d(int(in_channels / 2), in_channels, 1, 2)
            if is_not_first_layer
            else nn.Sequential()
        )

        return nn.ModuleDict(
            {"convert_block": convert_block, "residual_blocks": residual_blocks}
        )

    def residual_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> nn.Sequential:
        block_function = (
            bottleneck_residual_block if self.use_bottleneck else normal_residual_block
        )

        return block_function(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        last = None

        for layer in self.hidden_layers:
            convert_block = layer["convert_block"]
            residual_blocks = layer["residual_blocks"]

            for index, block in enumerate(residual_blocks):
                is_channel_changed = index == 1

                before = last
                last = x

                if before is not None:
                    if is_channel_changed:
                        before = convert_block(before)

                    x = torch.add(x, before)

                x = block(x)
                x = F.relu(x)

        x = self.output_block(x)

        return F.log_softmax(x, dim=0)
