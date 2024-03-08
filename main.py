import torch
from torch.utils.data import DataLoader

from src.models.res_net import ResNet
from src.data.celeb_a import CelebA

BATCH_SIZE = 5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(3, 1).to(device)

    dataset = CelebA()
    training_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    for item, target in training_loader:
        model(item)
        break

    pass


if __name__ == "__main__":
    main()
