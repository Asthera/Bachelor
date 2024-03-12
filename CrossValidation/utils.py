import os.path

import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from wandb.plot import confusion_matrix
from torchvision.io import read_image
import matplotlib.pyplot as plt


class FramesDataset(Dataset):

    # TODO: Check the resolution, we need it ?
    def __init__(self, frames_json, img_dir, transform=None, resolution=(224, 224)):
        self.transform = transform
        self.img_dir = img_dir
        self.all_frames = frames_json
        self.resolution = resolution

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        frame = self.all_frames[idx]

        frame_path = os.path.join(self.img_dir, frame['frame_cropped_path'])

        # read as tensor, torch.uint8
        image = read_image(frame_path)

        init_transform = transforms.Compose([
            # MUST BE

            transforms.Resize(self.resolution)
            # transforms.Pad([(pad_left, pad_right), (pad_top, pad_bottom)])

        ])

        image = init_transform(image)

        # convert to float, was uint8
        image = image.float()

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Here, you can access video_info if needed, e.g., video_info['name_cvat']
        # Example: Returning label, adjust as per your needs
        label = frame['label']  # Assuming label is at the video level
        image = image.expand(3, -1, -1)

        return image, label


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

    elif transform == "none" or transform is None:
        return None

    else:
        raise ValueError("Invalid transform")


def update_labels(frames_json):
    return [
        {**frame, 'label': 1 if frame['label'] == 2 else 0 if frame['label'] == 1 else frame['label']}
        for frame in frames_json
    ]


def build_dataset(batch_size, json_path, img_dir, transform=None):
    # TODO: add here checking for a, b line (now only b)

    train_transform = build_transforms(transform)

    # Load the JSON file
    data = json.load(open(json_path))

    # Get the frames from video(flattening) the JSON file
    train_frames_json = [frame for video in data for frame in video['frames_only_label'] if video['subset'] == 'train']
    test_frames_json = [frame for video in data for frame in video['frames_only_label'] if video['subset'] == 'test']

    show_sizes(train_frames_json, test_frames_json)

    # Split the json frames into train, validation, and test sets
    train_frames_json, val_frames_json = train_test_split(train_frames_json, test_size=0.2, random_state=42)

    # We have b_line, so need to change labels
    # 2 -> 1
    # 1 -> 0
    train_frames_json = update_labels(train_frames_json)
    val_frames_json = update_labels(val_frames_json)
    test_frames_json = update_labels(test_frames_json)

    # Create datasets
    train_dataset = FramesDataset(train_frames_json, img_dir, transform=train_transform)
    val_dataset = FramesDataset(val_frames_json, img_dir, transform=None)
    test_dataset = FramesDataset(test_frames_json, img_dir, transform=None)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, train_frames_json


def show_sizes(train_frames_json, test_frames_json):
    train_sizes = {}
    test_sizes = {}
    for frame in train_frames_json:
        # Show image
        plt.imshow(Image.open(frame['frame_cropped_path']))
        plt.axis('off')  # Optional: to hide the axis
        plt.show()

        init_transform = transforms.Compose([
            # MUST BE
            transforms.Resize((350, 224))
        ])
        plt.imshow(init_transform(Image.open(frame['frame_cropped_path'])))
        plt.axis('off')
        plt.show()


        if Image.open(frame['frame_cropped_path']).size not in train_sizes:
            train_sizes[Image.open(frame['frame_cropped_path']).size] = 1
        else:
            train_sizes[Image.open(frame['frame_cropped_path']).size] += 1

    for frame in test_frames_json:
        if Image.open(frame['frame_cropped_path']).size not in test_sizes:
            test_sizes[Image.open(frame['frame_cropped_path']).size] = 1
        else:
            test_sizes[Image.open(frame['frame_cropped_path']).size] += 1

    print("Train sizes: ", train_sizes)
    print("Train size sum: ", sum(train_sizes.values()))
    print()
    print("Test sizes: ", test_sizes)
    print("Test size sum: ", sum(test_sizes.values()))


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

        print(true_labels)
        print()
        print(predicted_labels)

        precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
        recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
        f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    return val_loss / len(val_loader), precision, recall, f1, balanced_accuracy

    # Log metrics using wandb here if needed


def test_model(network, test_loader, criterion, class_names, device="mps"):
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
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    confusion_matrix_wandb = confusion_matrix(probs=None,
                                              y_true=true_labels,
                                              preds=predicted_labels,
                                              class_names=class_names
                                              )

    return test_loss, test_accuracy, confusion_matrix_wandb, precision, recall, f1, balanced_accuracy
