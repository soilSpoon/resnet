import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.models.resnet import ResNet
from src.data.cifar import Cifar, CifarFileType

EPOCHS = 100
K = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001

CONFIG = {
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
}

num_layers, use_bottleneck = CONFIG[34]

model = ResNet(3, num_layers, use_bottleneck)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss().to(device)

    dataset = Cifar(
        root="/content/drive/MyDrive/FastCampus/data/cifar-10",
        file_type=CifarFileType.TRAIN,
        device=device,
    )

    training_dataset, validation_dataset = random_split(dataset, [0.9, 0.1])
    test_dataset = Cifar(
        root="/content/drive/MyDrive/FastCampus/data/cifar-10",
        file_type=CifarFileType.TEST,
        device=device,
    )

    training_loader = DataLoader(
        dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for item, label in training_loader:
            predictions = model(item)
            loss = criterion(predictions, label)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, training, loss: {total_loss / len(training_loader)}")

        model.eval()

        with torch.no_grad():
            total_loss = 0

            for item, label in validation_loader:
                predictions = model(item)
                loss = criterion(predictions, label)

                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}, validation, loss: {total_loss / len(validation_loader)}"
            )

    # pass


if __name__ == "__main__":
    main()
