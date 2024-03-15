import os.path
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import wandb
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np



class FramesDataset(Dataset):

    # TODO: Check the resolution, we need it ?
    def __init__(self, frames_json, img_dir, transform=None, resolution=None):
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


def build_transforms(transform_name):
    # Mapping from transform names to actual transformations
    transform_map = {
        "random_erasing_delete": transforms.RandomErasing(value=0),
        "random_erasing_random": transforms.RandomErasing(value="random"),
        "random_crop": transforms.RandomCrop(size=(200, 200)),
        "random_horizontal_flip": transforms.RandomHorizontalFlip(),
        "random_vertical_flip": transforms.RandomVerticalFlip(),
        "random_rotation": transforms.RandomRotation(degrees=70),
        "color_jitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        "random_noise(0.4, 0.5)": RandomNoise(p=0.5, mean=0.4, std=0.5),
        "random_noise(0, 0.1)": RandomNoise(p=0.5, mean=0, std=0.1),
        "none": None
    }

    if transform_name in transform_map:
        return transform_map[transform_name]
    else:
        raise ValueError(f"Invalid transform: {transform_name}")


def update_labels(frames_json):
    return [
        {**frame, 'label': 1 if frame['label'] == 2 else 0 if frame['label'] == 1 else frame['label']}
        for frame in frames_json
    ]


def build_dataset(batch_size, json_path, img_dir, transform=None, image_size=None):
    # TODO: add here checking for a, b line (now only b)

    if image_size is None:
        raise ValueError("Image size (height, width) must be provided")

    train_transform = build_transforms(transform)

    # Load the JSON file
    data = json.load(open(json_path))

    # Get the frames from video(flattening) the JSON file
    train_frames_json = [frame for video in data for frame in video['frames_only_label'] if video['subset'] == 'train']
    test_frames_json = [frame for video in data for frame in video['frames_only_label'] if video['subset'] == 'test']

    # show_sizes(train_frames_json, test_frames_json)

    # Split the json frames into train, validation, and test sets
    train_frames_json, val_frames_json = train_test_split(train_frames_json, test_size=0.2, random_state=42)

    # We have b_line, so need to change labels
    # 2 -> 1
    # 1 -> 0
    train_frames_json = update_labels(train_frames_json)
    val_frames_json = update_labels(val_frames_json)
    test_frames_json = update_labels(test_frames_json)

    # Create datasets
    train_dataset = FramesDataset(train_frames_json, img_dir, transform=train_transform, resolution=image_size)
    val_dataset = FramesDataset(val_frames_json, img_dir, transform=None, resolution=image_size)
    test_dataset = FramesDataset(test_frames_json, img_dir, transform=None, resolution=image_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


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


def build_optimizer(network, optimizer_name, learning_rate):
    if optimizer_name == "adam":
        return torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(network.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")


def build_criterion(criterion_name):
    if criterion_name == "cross-entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid criterion")


def build_network(fc_layer_size=512, number_of_classes=2, device="mps"):
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(fc_layer_size, number_of_classes)
    return model.to(device)


def train_epoch(network, train_loader, val_loader, optimizer, criterion, device="mps", epoch=0):
    network.train()

    train_loss = 0
    predicted_labels = []
    true_labels = []

    # Training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(images)

        _, predicted = outputs.max(1)

        loss = criterion(outputs, labels)
        train_loss += loss

        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    train_precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    train_recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    train_f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    train_balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    train_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
    train_confusion_matrix = train_confusion_matrix.reshape(1, -1)
    train_confusion_matrix = np.insert(train_confusion_matrix, 0, [epoch], axis=1)

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

        val_loss = val_loss / len(val_loader)
        val_precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
        val_recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
        val_f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
        val_balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

        val_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
        val_confusion_matrix = val_confusion_matrix.reshape(1, -1)
        val_confusion_matrix = np.insert(val_confusion_matrix, 0, [epoch], axis=1)

        print("Val True labels: ", true_labels)
        print("Val Predicted labels: ", predicted_labels)


    return (train_loss, train_precision, train_recall, train_f1, train_balanced_accuracy, train_confusion_matrix,
            val_loss, val_precision, val_recall, val_f1, val_balanced_accuracy, val_confusion_matrix)


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




    confusion_matrix_wandb = wandb.plot.confusion_matrix(probs=None,
                                              y_true=true_labels,
                                              preds=predicted_labels,
                                              class_names=class_names
                                              )

    return test_loss, test_accuracy, confusion_matrix_wandb, precision, recall, f1, balanced_accuracy
