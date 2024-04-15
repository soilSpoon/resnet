import os
from pathlib import Path
from typing import Final, Literal

import lightning as L
from beartype import beartype
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.celeb_a import CelebA
from src.models.resnet import ResNet


@beartype
class Action:

    def __init__(
        self,
        model: ResNet,
        logger: Logger,
        mode: Literal["train", "test"] = "train",
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience=3,
    ) -> None:
        self.model = model
        self.logger = logger

        self.mode: Final = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

    def __call__(self) -> None:
        if self.mode == "train":
            self._train()
        else:
            self._test()

    def create_dataloader(self, dataset: Dataset, shuffle=False):
        num_workers = os.cpu_count()

        if num_workers is None:
            num_workers = 1

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    @property
    def output_path(self):
        path = os.getenv("OUTPUT_PATH")

        if path is None:
            raise ValueError("Environment variable 'OUTPUT_PATH' is not set.")

        return path

    @property
    def pre_text(self) -> str:
        return f"{self.model.model_size}-{self.epochs}-{self.batch_size}-{self.early_stopping_patience}"

    @property
    def model_checkpoint_dir(self):
        return f"{self.output_path}/checkpoints/{self.pre_text}"

    @property
    def trainer(self) -> L.Trainer:
        return L.Trainer(
            logger=self.logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.model_checkpoint_dir,
                    filename=f"{{epoch}}-{{val_f1:.4f}}",
                    save_top_k=self.early_stopping_patience,
                    monitor="val_f1",
                    mode="max",
                ),
                RichModelSummary(max_depth=-1),
                EarlyStopping(
                    monitor="val_f1", mode="max", patience=self.early_stopping_patience
                ),
            ],
            enable_model_summary=False,
            limit_train_batches=self.batch_size,
            max_epochs=self.epochs,
        )

    def _train(self) -> None:
        train_dataset, validation_dataset = random_split(CelebA("train"), [0.9, 0.1])

        train_loader = self.create_dataloader(train_dataset)
        validation_loader = self.create_dataloader(validation_dataset)

        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

    def _test(self) -> None:
        test_loader = self.create_dataloader(CelebA("test"))

        checkpoints = sorted(Path(self.output_path).glob("**/*.ckpt"))

        if len(checkpoints) == 0:
            raise FileNotFoundError("No checkpoint found for testing.")

        *_, last_ckpt = checkpoints

        self.model = ResNet.load_from_checkpoint(
            last_ckpt,
            input_size=self.model.input_size,
            num_layers=self.model.num_layers,
            num_classes=self.model.num_classes,
            use_bottleneck=self.model.use_bottleneck,
        )

        self.trainer.test(self.model, dataloaders=test_loader)
