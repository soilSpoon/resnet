import os
from typing import Literal, Tuple
import pandas as pd

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CelebA(Dataset):
    def __init__(self, mode: Literal["train", "test"] = "train"):
        self.mode = mode

        attr: pd.DataFrame = pd.read_csv(
            f"{self.root}/Anno/{self.filename}", index_col=0
        )

        # map from {-1, 1} to {0, 1}
        for column in attr.columns:
            attr[column] = attr[column].replace(-1, 0)

        self.transform = v2.Compose(
            [
                v2.CenterCrop((178, 178)),
                # v2.Resize((128, 128)),
                v2.PILToTensor(),
                v2.ToDtype(dtype=torch.float32, scale=True),
            ]
        )

        self.filenames = attr.index
        self.attr = attr
        self.attr_names = attr.columns

    @property
    def root(self) -> str:
        path = os.getenv("DATA_PATH")

        if path is None:
            raise Exception("Environment variable 'DATA_PATH' is not set.")

        return path

    @property
    def filename(self) -> Literal["train.csv", "test.csv"]:
        return "train.csv" if self.mode else "test.csv"

    def __len__(self) -> int:
        return len(self.attr)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = PIL.Image.open(
            f"{self.root}/Img/img_align_celeba/{self.filenames[index]}"
        )

        X: torch.Tensor = self.transform(img)
        target = self.attr.iloc[index]

        return X, torch.Tensor(target.values)
