import os

import torch
import wandb

from utils import build_dataset, build_network, build_optimizer, train_epoch, test_model
import yaml


def train():
    # Initialize a new wandb run
    with open("sweep.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # giving the run a name

    custom_name = f"tr:{run.config.transform}_{run.config.fold}_batch:{run.config.batch_size}_{run.id}"

    wandb.run.name = custom_name
    # wandb.config.fold = "fold_0"
    folds_path = "metadata_folds/kkui-lung-bline-lumify:latest/video"
    fold_json_path = os.path.join(folds_path, wandb.config.fold + ".json")
    img_dir = ""

    train_loader, val_loader, test_loader = build_dataset(wandb.config.batch_size, fold_json_path,
                                                          img_dir,
                                                          wandb.config.transform)

    fc_layer_size = 512
    number_of_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    network = build_network(fc_layer_size, number_of_classes, device)

    optimizer = "adam"
    learning_rate = 0.001
    epochs = 19
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(network, optimizer, learning_rate)

    for epoch in range(epochs):
        val_loss, precision, recall, f1, balanced_accuracy = train_epoch(network, train_loader, val_loader, optimizer, criterion, device)

        wandb.log({"val_loss": val_loss, "val_precision": precision, "val_recall": recall, "val_f1": f1, "val_balanced_acc": balanced_accuracy})
    # Test model
    class_names = ["0", "1"]

    test_loss, test_accuracy, confusion_matrix, precision, recall, f1, balanced_accuracy = test_model(network, test_loader,
                                                                                                criterion, class_names, device)

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": confusion_matrix,
        "test_balanced_acc": balanced_accuracy
    })


train()
