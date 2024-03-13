from typing import List, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


def conv(
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Module:
    padding = int((kernel_size - 1) / 2)

    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(out_channels),
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    return conv(1, in_channels, out_channels, stride=stride)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return conv(3, in_channels, out_channels, stride=stride)


def conv7x7(in_channels: int, out_channels: int, stride: int = 1):
    return conv(7, in_channels, out_channels, stride=stride)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = conv1x1(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if x.shape != identity.shape:
            identity = self.downsample(identity)
        x += identity
        x = F.relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        middle_channels = int(out_channels / 4)

        self.conv1 = conv1x1(in_channels, middle_channels, stride=stride)
        self.conv2 = conv3x3(middle_channels, middle_channels)
        self.conv3 = conv1x1(middle_channels, out_channels)
        self.downsample = conv1x1(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        if x.shape != identity.shape:
            identity = self.downsample(identity)
        x += identity
        x = F.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, input_size: int, num_layers: List[int], use_bottleneck: bool):
        super().__init__()

        self.use_bottleneck = use_bottleneck
        last_out_channels_weight = 4 if use_bottleneck else 1
        self.last_out_channels = (
            64 * (2 ** (len(num_layers) - 1)) * last_out_channels_weight
        )
        self.residual_block = Bottleneck if self.use_bottleneck else BasicBlock

        self.input_layer = self.input_block(input_size)
        self.output_layer = self.output_block()
        self.hidden_layers = nn.ModuleList([])

        for layer_index, num_layer in enumerate(num_layers):
            hidden_layer = self.hidden_layer(num_layer, layer_index)
            self.hidden_layers.append(hidden_layer)

    def input_block(self, input_size: int):
        return nn.Sequential(
            conv7x7(input_size, 64, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

    def output_block(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.last_out_channels, 1000),
            nn.Linear(1000, 10),
        )

    def hidden_layer(
        self, residual_block_count: int, layer_index: int
    ) -> nn.ModuleDict:
        is_first_layer = layer_index == 0

        in_channels = 64 * 2**layer_index
        out_channels_weight = 4 if self.use_bottleneck else 1
        out_channels = in_channels * out_channels_weight

        residual_blocks = nn.ModuleList([])

        for block_index in range(residual_block_count):
            is_first_block = block_index == 0

            stride = 1
            in_channels_weight = 1

            if not is_first_layer and is_first_block:
                stride *= 2
                in_channels_weight *= 0.5

            if self.use_bottleneck and not (is_first_layer and is_first_block):
                in_channels_weight *= 4

            residual_block = self.residual_block(
                int(in_channels * in_channels_weight), out_channels, stride
            )

            residual_blocks.append(residual_block)

        return residual_blocks

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)

        for residual_blocks in self.hidden_layers:
            for block in residual_blocks:
                x = block(x)

        x = self.output_layer(x)

        return F.log_softmax(x, dim=0)
