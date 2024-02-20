import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from utils import CustomDataset  # Ensure this includes necessary transformations
import pandas as pd
from sklearn.model_selection import train_test_split


def build_transforms(transform):
    if transform == "random_erasing":
        return transforms.RandomErasing()

    elif transform == "random_crop":
        return transforms.RandomCrop(size=(200, 200))

    elif transform == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip()

    elif transform == "random_vertical_flip":
        return transforms.RandomVerticalFlip()

    elif transform == "random_rotation":
        return transforms.RandomRotation(degrees=90)

    elif transform == "none":
        return None

    else:
        raise ValueError("Invalid transform")


def build_dataset(batch_size, data_path, train_dataset_path, test_dataset_path, transform=None):
    train_transform = build_transforms(transform)

    # Load dataframes
    train_dataframe = pd.read_csv(train_dataset_path)
    test_dataframe = pd.read_csv(test_dataset_path)

    # Filter out specific rows if needed
    train_dataframe = train_dataframe.loc[train_dataframe["name_cvat"] != "021_image_182041262860754.mp4"]

    # Split train dataframe into train and validation
    train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = CustomDataset(train_dataframe, data_path, transform=train_transform)
    val_dataset = CustomDataset(val_dataframe, data_path, transform=None)
    test_dataset = CustomDataset(test_dataframe, data_path, transform=None)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "adam":
        return torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(network.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")


def build_network(fc_layer_size=512, number_of_classes=2, device="mps"):
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(fc_layer_size, number_of_classes)
    return model.to(device)


def train_epoch(network, train_loader, val_loader, optimizer, criterion, device="mps"):
    network.train()
    # Training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    network.eval()
    # Validation loop
    with torch.no_grad():
        val_loss = 0

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            val_loss += criterion(outputs, labels).item()

            # Calculate validation loss and accuracy

    return val_loss / len(val_loader)

    # Log metrics using wandb here if needed


def test_model(network, test_loader, criterion, device="mps"):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * (correct / total)
    test_loss = test_loss / len(test_loader)

    return test_loss, test_accuracy
