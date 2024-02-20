import torch
import wandb
from utilsweep import build_dataset, build_network, build_optimizer, train_epoch, test_model
import yaml


def train():
    # Initialize a new wandb run
    with open("sweep_grid.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # giving the run a name

    custom_name = f"tr-{run.config.transform}-{run.config.batch_size}-{run.id}"

    wandb.run.name = custom_name

    data_path = "/Volumes/LaCie/Macbook/"
    train_dataset_path = "/Volumes/LaCie/Macbook/datasets/29DVOHC6Q45Q_train.csv"
    test_dataset_path = "/Volumes/LaCie/Macbook/datasets/29DVOHC6Q45Q_test.csv"

    train_loader, val_loader, test_loader = build_dataset(wandb.config.batch_size, data_path,
                                                          train_dataset_path, test_dataset_path,
                                                          wandb.config.transform)

    fc_layer_size = 512
    number_of_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    network = build_network(fc_layer_size, number_of_classes, device)

    optimizer = "adam"
    learning_rate = 0.001
    epochs = 8
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(network, optimizer, learning_rate)

    for epoch in range(epochs):
        val_loss = train_epoch(network, train_loader, val_loader, optimizer, criterion, device)
        wandb.log({"val_loss": val_loss})

    # Test model
    test_loss, test_accuracy = test_model(network, test_loader, criterion, device)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})


train()
