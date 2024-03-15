import os

import torch
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint

from argparse import ArgumentParser, Namespace

from src.models.resnet import ResNet
from src.data.cifar import Cifar, CifarFileType

WANDB_PATH = "/content/wandb"

EPOCHS = 100
K = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001

CONFIG = {
    0: ([1, 1], False),
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
}

NUM_WORKERS = os.cpu_count()

DATA_LOADER_PARAMS = {
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "num_workers": NUM_WORKERS,
}

INPUT_CHANNELS = 3
NUM_CLASSES = 10


def train(model: ResNet, trainer: L.Trainer):
    # dataset = CelebA()
    dataset = Cifar(file_type=CifarFileType.TRAIN)

    train_dataset, validation_dataset = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(dataset=train_dataset, **DATA_LOADER_PARAMS)
    validation_loader = DataLoader(dataset=validation_dataset, **DATA_LOADER_PARAMS)

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )


def test(model: ResNet, trainer: L.Trainer):
    test_dataset = Cifar(file_type=CifarFileType.TEST)

    test_loader = DataLoader(dataset=test_dataset, **DATA_LOADER_PARAMS)

    trainer.test(model, dataloaders=test_loader)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train")
    group.add_argument("--test")

    parser.add_argument("--size", choices=[0, 18, 34, 50, 101, 152], default=0)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    num_layers, use_bottleneck = CONFIG[args.size]

    model = ResNet(
        INPUT_CHANNELS,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        use_bottleneck=use_bottleneck,
        example_input_array=torch.Tensor(256, 3, 32, 32),
    )

    trainer = L.Trainer(
        logger=WandbLogger(save_dir=WANDB_PATH),
        callbacks=[
            ModelCheckpoint(),
            ModelSummary(max_depth=-1),
            EarlyStopping(monitor="val_f1", mode="max"),
        ],
        limit_train_batches=BATCH_SIZE,
        max_epochs=EPOCHS,
    )

    if args.train:
        train(model, trainer)
    else:
        test(model, trainer)


if __name__ == "__main__":
    main()
