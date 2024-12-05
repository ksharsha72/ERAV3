import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    # Initialize model
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    total_batches = len(train_loader)
    print(f"\nStarting training for 1 epoch...")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            progress = 100.0 * batch_idx / total_batches
            print(
                f"Progress: {progress:.1f}% [{batch_idx}/{total_batches}] Loss: {loss.item():.4f}"
            )

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_mnist_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    train()
