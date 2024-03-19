import numpy as np
import pandas as pd

from typing import Optional
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = "/content/drive/MyDrive/FastCampus/data/CelebA"
ROOT = "/teamspace/studios/this_studio/data/CelebA"


class CelebA(Dataset):

    def __init__(self):
        # map from {-1, 1} to {0, 1}
        attr = pd.read_csv(f"{ROOT}/Anno/list_attr_celeba.txt", sep="\s+", skiprows=1)

        for col in attr.columns[(attr == 1).any() & (attr == -1).any()]:
            attr[col] = attr[col].replace(-1, 0)

        self.transform = v2.Compose([
            v2.CenterCrop((178, 178)),
            # v2.Resize((128, 128)),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])

        self.filename = attr.index
        self.attr = attr
        self.attr_names = attr.columns

        print(self.attr_names)

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, index):
        img = PIL.Image.open(f"{ROOT}/Img/img_align_celeba/{self.filename[index]}")

        X = self.transform(img)
        target = self.attr['Male'].iloc[index]

        return X, target
