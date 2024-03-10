import torch
from torch.utils.data import DataLoader

from src.models.resnet import ResNet
from src.data.cifar import Cifar, CifarFileType

BATCH_SIZE = 5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(3, 1).to(device)
    dataset = Cifar(
        root="/content/drive/MyDrive/FastCampus/data/cifar-10",
        file_type=CifarFileType.TRAIN,
        device=device,
    )

    training_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    for item, label in training_loader:
        print(item.shape)

        model(item)
        break

    # pass


if __name__ == "__main__":
    main()
