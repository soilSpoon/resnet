from collections import namedtuple
import csv
from typing import Optional
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(Dataset):

    def __init__(self, device: torch.device):
        splits = self._load_csv("list_eval_partition.txt")
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        self.device = device
        self.transform = transforms.ToTensor()

        self.filename = splits.index
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(attr.data + 1, 2, rounding_mode="floor").to(device)
        self.attr_names = attr.header

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, index):
        img = PIL.Image.open(f"/content/drive/MyDrive/FastCampus/data/Img/img_align_celeba/{self.filename[index]}")

        X = self.transform(img).to(self.device)
        target = self.attr[index, :]

        return X, target

    def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
        with open(f"/content/drive/MyDrive/FastCampus/data/Anno/{filename}") as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))
