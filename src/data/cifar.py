from enum import Enum

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CifarFileType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


class Cifar(Dataset):
    FILES = {
        CifarFileType.TRAIN: [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ],
        CifarFileType.TEST: ["test_batch"],
    }

    def __init__(self, root: str, file_type: CifarFileType):
        self.items, self.labels = self.read_items(root, file_type)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        label = self.labels[index]
        item = self.items[index]
        item = item / 255.0
        item = item

        return item, label

    def read_items(self, root: str, file_type: CifarFileType):
        files = Cifar.FILES[file_type]

        data = []
        labels = []

        for file in files:
            dict = self.unpickle(f"{root}/{file}")

            data.append(dict["data"])
            labels.extend(dict["labels"])

        data = np.vstack(data).reshape(-1, 3, 32, 32)

        return data, labels

    def unpickle(self, filepath: str):
        import pickle

        with open(filepath, "rb") as file:
            dict = pickle.load(file, encoding="latin1")

        return dict
