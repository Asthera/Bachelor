import os
import time

import torch
import wandb
from utils import build_dataset, build_network, build_optimizer, build_criterion, train_epoch, test_model
import yaml


def train():
    # Initialize a new wandb run
    with open("sweep.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # giving the run a name
    custom_name = f"tr:{run.config.transform}_{run.config.fold}_batch:{run.config.batch_size}_{run.id}"
    wandb.run.name = custom_name

    # for example = wandb.config.fold = "fold_0"
    fold_json_path = os.path.join(run.config.folds_path, wandb.config.fold + ".json")

    image_size = (run.config.image_size["height"], run.config.image_size["width"])
    train_loader, val_loader, test_loader = build_dataset(run.config.batch_size,
                                                          fold_json_path,
                                                          run.config.img_dir,
                                                          run.config.transform,
                                                          image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    network = build_network(run.config.fc_layer_size, run.config.number_of_classes, device)

    epochs = run.config.epochs

    criterion = build_criterion(run.config.criterion)
    optimizer = build_optimizer(network, run.config.optimizer, run.config.learning_rate)

    # Train model

    patience = run.config.patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state_dict = None

    train_confusion_matrix = wandb.Table(columns=["Step", "TP", "FP", "TN", "FN"])
    val_confusion_matrix = wandb.Table(columns=["Step", "TP", "FP", "TN", "FN"])

    early_stopping = False

    for epoch in range(epochs):

        start_time = time.time()
        # TODO: Split train_epoch into train and validate functions
        (train_loss, train_precision, train_recall, train_f1, train_balanced_accuracy, train_conf_matrix,
         val_loss, val_precision, val_recall, val_f1, val_balanced_accuracy, val_conf_matrix) = train_epoch(network,
                                                                                                            train_loader,
                                                                                                            val_loader,
                                                                                                            optimizer,
                                                                                                            criterion,
                                                                                                            device,
                                                                                                            epoch)

        # Check for early stopping

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # best_model_state_dict = network.state_dict()
            torch.save(network.state_dict(),
                       "weights/{run_id}_{epoch}_{val_bal_acc}_{val_recall}_{val_loss}_{transform}.pth".format(
                           run_id=run.id,
                           epoch=epoch,
                           val_bal_acc=val_balanced_accuracy,
                           val_recall=val_recall,
                           val_loss=val_loss,
                           transform=run.config.transform))
            print(
                f"Model saved at epoch {epoch} with val_loss: {val_loss}, val_balanced_accuracy: {val_balanced_accuracy}, val_recall: {val_recall} transformation: {run.config.transform}")
        else:
            epochs_without_improvement += 1

            print(f"Early stopping counter: {epochs_without_improvement} / {patience} {run.id}")

            # TODO: Add here ability to fine-tining
            # if fine_tune_enable:

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} with val_loss: {val_loss}, val_balanced_accuracy: {val_balanced_accuracy}, val_recall: {val_recall} transformation: {run.config.transform}")
                wandb.run.summary["stopping"] = epoch
                early_stopping = True

        val_confusion_matrix.add_data(*val_conf_matrix[0])
        train_confusion_matrix.add_data(*train_conf_matrix[0])


        wandb.log({"val_loss": val_loss, "val_precision": val_precision,
                   "val_recall": val_recall, "val_f1": val_f1,
                   "val_balanced_acc": val_balanced_accuracy,

                   "train_loss": train_loss, "train_precision": train_precision,
                   "train_recall": train_recall, "train_f1": train_f1,
                   "train_balanced_acc": train_balanced_accuracy})

        print(f"Epoch {epoch} took {time.time() - start_time} seconds")

        if early_stopping:
            break

    wandb.log({"vaL_confusion_matrix": val_confusion_matrix, "train_confusion_matrix": train_confusion_matrix})

    # Test model
    class_names = ["0", "1"]

    (test_loss, test_accuracy, test_confusion_matrix, test_precision,
     test_recall, test_f1, test_balanced_accuracy) = test_model(network,
                                                                test_loader,
                                                                criterion,
                                                                class_names,
                                                                device)

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "confusion_matrix": test_confusion_matrix,
        "test_balanced_acc": test_balanced_accuracy
    })


train()
