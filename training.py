import time

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import CustomDataset
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="bachelor-test",
    name="no_aug_40epochs",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "resnet18",
        "epochs": 40,
        "batch_size": 32,
        "dataset": "29DVOHC6Q45Q",
        "pretrained_model": True
    }
)

data_path = "/Volumes/LaCie/Macbook/"
train_dataset_path = "/Volumes/LaCie/Macbook/datasets/29DVOHC6Q45Q_train.csv"
test_dataset_path = "/Volumes/LaCie/Macbook/datasets/29DVOHC6Q45Q_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
number_of_classes = 2

# Hyperparameters
learning_rate = 0.001
pretrained_model = True
batch_size = 32
epochs = 4

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, number_of_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load data

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomErasing()

])

# dataframes
train_dataframe = pd.read_csv(train_dataset_path)
test_dataframe = pd.read_csv(test_dataset_path)

train_dataframe = train_dataframe.loc[train_dataframe["name_cvat"] != "021_image_182041262860754.mp4"]

# separate train dataframe to train and validation
train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.2, random_state=42)

print("Count of dataframes train, val, test")
print(len(train_dataframe))
print(len(val_dataframe))
print(len(test_dataframe))

# datasets
train_dataset = CustomDataset(train_dataframe, data_path, transform=None)
val_dataset = CustomDataset(val_dataframe, data_path, transform=None)
test_dataset = CustomDataset(test_dataframe, data_path, transform=None)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train model
time_total = 0

for epoch in range(epochs):

    start_time = time.time()

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        # .item() to return the value of the tensor
        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy_epoch = (train_correct / train_total) * 100
    train_loss_epoch = train_loss / len(train_loader)

    # Validation

    model.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0
        val_total = 0

        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_accuracy_epoch = (val_correct / val_total) * 100
        val_loss_epoch = val_loss / len(val_loader)

    end_time = time.time()
    time_total += end_time - start_time
    print("Epoch time: ", end_time - start_time)

    wandb.log({
        "Train Loss": train_loss_epoch,
        "Train Accuracy": train_accuracy_epoch,
        "Val Loss": val_loss_epoch,
        "Val Accuracy": val_accuracy_epoch
    })

    print(f"Epoch {epoch+1}/{epochs} Train Accuracy: "
          f"{train_accuracy_epoch:.2f}% Val Accuracy: {val_accuracy_epoch:.2f}%")

print("Average time: ", time_total / epochs)
print("Total time: ", time_total)


# Test model

model.eval()

with torch.no_grad():
    test_loss = 0
    test_correct = 0
    test_total = 0

    true_labels = []
    predicted_labels = []

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)

        predicted_labels.extend(predicted.item() for predicted in predicted)
        true_labels.extend(label.item() for label in labels)

        test_loss += loss.item()
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_accuracy = (test_correct / test_total) * 100
    test_loss = test_loss / len(test_loader)

    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

    # log confusion matrix
    class_names = ["0", "1"]

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_labels,
            preds=predicted_labels,
            class_names=class_names
        )
    })
