from typing import List, Tuple, Union

from torch import optim, nn, Tensor
from torch.nn import functional as F
from torchmetrics.classification import MulticlassF1Score

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
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
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


class ResNet(L.LightningModule):
    INITIAL_CHANNELS = 64

    def __init__(
        self,
        input_size: int,
        num_layers: List[int],
        num_classes: int,
        use_bottleneck: bool,
        example_input_array: Union[Tensor, None] = None,
    ):
        super().__init__()

        if example_input_array is not None:
            self.example_input_array = example_input_array

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # bottlenck 사용 여부에 따라 ouput channels 조정
        self.block_weight = 4 if use_bottleneck else 1

        self.last_out_channels = (
            ResNet.INITIAL_CHANNELS * (2 ** (len(num_layers) - 1)) * self.block_weight
        )
        self.residual_block = Bottleneck if use_bottleneck else BasicBlock

        self.input_layer = self.input_block(input_size)
        self.output_layer = self.output_block(num_classes)
        self.hidden_layers = nn.ModuleList([])

        for layer_index, num_layer in enumerate(num_layers):
            hidden_layer = self.hidden_layer(num_layer, layer_index)
            self.hidden_layers.append(hidden_layer)

    def input_block(self, input_size: int):
        return nn.Sequential(
            conv7x7(input_size, ResNet.INITIAL_CHANNELS, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

    def output_block(self, num_classes: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.last_out_channels, 1000),
            nn.Linear(1000, num_classes),
        )

    def hidden_layer(
        self, residual_block_count: int, layer_index: int
    ) -> nn.ModuleDict:
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

        for residual_blocks in self.hidden_layers:
            for block in residual_blocks:
                x = block(x)

        x = self.output_layer(x)

        return F.log_softmax(x, dim=0)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.nll_loss(predictions, targets)
        self.train_f1(predictions, targets)

        self.log_dict(
            {"train_loss": loss, "train_f1": self.train_f1},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.nll_loss(predictions, targets)
        self.val_f1(predictions, targets)

        self.log_dict(
            {"val_loss": loss, "val_f1": self.val_f1},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        inputs, targets = batch
        predictions = self(inputs)

        loss = F.nll_loss(predictions, targets)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
