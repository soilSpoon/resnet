import os
from typing import Final, Literal
from beartype import beartype

from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary

from src.data.celeb_a import CelebA
from src.models.resnet import ResNet


@beartype
class Action:
    def __init__(
        self,
        model: ResNet,
        mode: Literal["train", "test"] = "train",
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience=3,
    ) -> None:
        self.model = model
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
            logger=WandbLogger(
                project="resnet",
                name=f"{self.mode}-{self.pre_text}",
                save_dir=self.output_path,
            ),
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

        # *_, last_ckpt = sorted(outdir.glob("**/*.ckpt"))

        # self.model.load_from_checkpoint()

        self.trainer.test(self.model, dataloaders=test_loader)
