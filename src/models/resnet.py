from typing import List, Tuple

import lightning as L
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.classification import MulticlassF1Score


def conv(
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Module:
    padding = int((kernel_size - 1) / 2)

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return conv(1, in_channels, out_channels, stride=stride)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return conv(3, in_channels, out_channels, stride=stride)


def conv7x7(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
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


class ResNet(L.LightningModule):
    INITIAL_CHANNELS = 64

    def __init__(
        self,
        input_size: int,
        num_layers: List[int],
        num_classes: int,
        use_bottleneck: bool,
    ):
        super().__init__()

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.input_size = input_size
        self.use_bottleneck = use_bottleneck
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.residual_block = Bottleneck if use_bottleneck else BasicBlock

        self.input_layer = self.input_block(input_size)
        self.output_layer = self.output_block(num_classes)
        self.hidden_layers = nn.ModuleList([])

        for layer_index, num_layer in enumerate(num_layers):
            hidden_layer = self.hidden_layer(num_layer, layer_index)
            self.hidden_layers.append(hidden_layer)

        self.initialize_weights()

    @property
    def last_out_channels(self):
        return (
            ResNet.INITIAL_CHANNELS
            * (2 ** (len(self.num_layers) - 1))
            * self.block_weight
        )

    @property
    def model_size(self):
        return sum(self.num_layers) * (3 if self.use_bottleneck else 2) + 2

    @property
    def block_weight(self):
        # bottlenck 사용 여부에 따라 ouput channels 조정
        return 4 if self.use_bottleneck else 1

    @property
    def example_input_array(self) -> torch.Tensor:
        return torch.Tensor(256, 3, 128, 128)

    def input_block(self, input_size: int) -> nn.Module:
        return nn.Sequential(
            conv7x7(input_size, ResNet.INITIAL_CHANNELS, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

    def output_block(self, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.last_out_channels, 1000),
            nn.Linear(1000, num_classes),
        )

    def hidden_layer(
        self, residual_block_count: int, layer_index: int
    ) -> nn.ModuleList:
        is_first_layer = layer_index == 0

        base_in_channels = ResNet.INITIAL_CHANNELS * 2**layer_index
        out_channels_weight = self.block_weight
        out_channels = base_in_channels * out_channels_weight

        residual_blocks = nn.ModuleList([])

        for block_index in range(residual_block_count):
            is_first_block = block_index == 0

            stride = 1
            in_channels_weight = 1

            # 레이어 바뀔 때 downsample 인자 처리
            if not is_first_layer and is_first_block:
                stride *= 2
                in_channels_weight *= 0.5

            # basic, bottleneck weight 처리
            if not (is_first_layer and is_first_block):
                in_channels_weight *= self.block_weight

            in_channels = int(base_in_channels * in_channels_weight)

            residual_block = self.residual_block(in_channels, out_channels, stride)

            residual_blocks.append(residual_block)

        return residual_blocks

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)

        residual_blocks: nn.ModuleList

        for residual_blocks in self.hidden_layers:  # type: ignore
            for block in residual_blocks:
                x = block(x)

        x = self.output_layer(x)

        return F.sigmoid(x)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        self.train_f1(predictions, targets)

        self.log_dict(
            {"train_loss": loss, "train_f1": self.train_f1},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        self.val_f1(predictions, targets)

        self.log_dict(
            {"val_loss": loss, "val_f1": self.val_f1},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.binary_cross_entropy_with_logits(predictions, targets)

        self.log("test_loss", loss)

    def configure_optimizers(self) -> optim.Adam:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def initialize_weights(self):
        if isinstance(self, nn.Conv2d):
            nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(self, nn.BatchNorm2d):
            self.weight.data.fill_(1)
            self.bias.data.zero_()
        elif isinstance(self, nn.Linear):
            nn.init.kaiming_normal_(self.weight, mode="fan_avg", nonlinearity="relu")
