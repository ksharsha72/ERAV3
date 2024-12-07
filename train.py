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

    # Enhanced data augmentation pipeline
    transform_train = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,  # Random rotation up to 10 degrees
                translate=(0.1, 0.1),  # Random translation up to 10%
                scale=(0.9, 1.1),  # Random scaling between 90% and 110%
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Random erasing
        ]
    )

    # Test transform should remain simple
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset with augmentation
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform_train),
        batch_size=64,
        shuffle=True,
    )

    # Initialize model
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    total_batches = len(train_loader)
    print(f"\nStarting training for 1 epoch...")
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy during training
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            progress = 100.0 * batch_idx / total_batches
            current_acc = 100.0 * correct / total if total > 0 else 0
            print(
                f"Progress: {progress:.1f}% [{batch_idx}/{total_batches}] "
                f"Loss: {running_loss/(batch_idx+1):.4f} "
                f"Training Accuracy: {current_acc:.2f}%"
            )

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_mnist_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    train()
