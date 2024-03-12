import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

from torchvision.models import ResNet18_Weights
from utils import CustomDataset  # Ensure this includes necessary transformations
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import transforms
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score


class RandomSpeckleNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        super().__init__()
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if torch.rand(1) < self.p:
            noise = torch.randn(img.size()) * self.std + self.mean
            img = img + img * noise
        return img


class RandomNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if torch.rand(1) < self.p:
            noise = torch.randn(img.size()) * self.std + self.mean
            img_tensor = torch.clamp(img + noise, 0, 1)

            return img_tensor
        return img


def build_transforms(transform):
    if transform == "random_erasing_delete":
        return transforms.RandomErasing(value=0)

    elif transform == "random_erasing_random":
        return transforms.RandomErasing(value="random")

    elif transform == "random_crop":
        return transforms.RandomCrop(size=(200, 200))

    elif transform == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip()

    elif transform == "random_vertical_flip":
        return transforms.RandomVerticalFlip()

    elif transform == "random_rotation":
        return transforms.RandomRotation(degrees=70)

    elif transform == "color_jitter":
        "Randomly change the brightness, contrast, saturation and hue of an image."
        return transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    elif transform == "random_speckle_noise":
        "Not implemented, because it is not undersatood how to works params, too strong noise"
        return NotImplementedError()

    elif transform == "random_noise(0.4, 0.5)":
        return RandomNoise(p=0.5, mean=0.4, std=0.5)

    elif transform == "random_noise(0, 0.1)":
        return RandomNoise(p=0.5, mean=0, std=0.1)

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

        predicted_labels = []
        true_labels = []

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            val_loss += criterion(outputs, labels).item()

            _, predicted = outputs.max(1)

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
        recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
        f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)

    return val_loss / len(val_loader), precision, recall, f1

    # Log metrics using wandb here if needed


def test_model(network, test_loader, criterion, device="mps"):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)

            predicted_labels.extend(predicted.item() for predicted in predicted)
            true_labels.extend(label.item() for label in labels)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * (correct / total)
    test_loss = test_loss / len(test_loader)

    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)

    return test_loss, test_accuracy, true_labels, predicted_labels, precision, recall, f1
