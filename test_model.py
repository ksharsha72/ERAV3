import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import glob
import pytest


def get_latest_model():
    model_files = glob.glob("model_mnist_*.pth")
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model


def test_model_architecture():
    model = MNISTModel()

    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 25000
    ), f"Model has {total_params} parameters, should be less than 25000"


def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)

    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test transform without augmentation
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=transform_test),
        batch_size=1000,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nModel Accuracy: {accuracy:.2f}%")
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"


if __name__ == "__main__":
    pytest.main([__file__])
