from datetime import datetime
from torch import no_grad, save
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from wandb import Table
from torch import device as torch_device
from copy import copy
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, network, train_loader, val_loader, test_loader, optimizer, criterion, run, device_name, patience=-1,
                 early_stopping: bool = True or False, save_weights: bool = True or False, test_with_best_val_loss: bool = True or False):

        self.epochs_with_best_val_loss = None
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.run = run
        self.device = torch_device(device_name)
        self.patience = patience
        self.early_stopping = early_stopping

        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        self.train_confusion_matrix = Table(columns=["TN", "FP", "FN", "TP"])
        self.val_confusion_matrix = Table(columns=["TN", "FP", "FN", "TP"])

        # was removed to send test to wandb part because not sending to wandb
        # self.test_confusion_matrix = Table(columns=["TN", "FP", "FN", "TP"])

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state_dict = None
        self.save_weights = save_weights

        self.test_with_best_val_loss = test_with_best_val_loss

        self.current_epoch = 0


    def show_images(self, images, labels):

        batch = images.shape[0]
        fig, axs = plt.subplots(1, batch, figsize=(30, 30))

        for i in range(batch):
            axs[i].imshow(images[i].permute(1, 2, 0).int(), cmap='gray')
            axs[i].set_title(labels[i].item())
            axs[i].axis('off')

        plt.show()

        # close when W pressed
        plt.waitforbuttonpress()

    def train_epoch(self):
        self.network.train()
        train_loss = 0.0
        predicted_labels = []
        true_labels = []

        for images, labels in self.train_loader:

            # show images and labels
            # self.show_images(images, labels)

            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.network(images)
            _, predicted = outputs.max(1)
            loss = self.criterion(outputs, labels)
            train_loss += loss.item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            loss.backward()
            self.optimizer.step()

        print(f"train part({self.current_epoch}ep):")
        print("predicted_labels, true_labels")
        print(predicted_labels, true_labels)
        print()



        self.train_metrics = self.calculate_metrics(true_labels, predicted_labels) + (train_loss,)

        self.current_epoch += 1

    def validate(self):
        self.network.eval()
        val_loss = 0.0
        predicted_labels = []
        true_labels = []

        with no_grad():
            for images, labels in self.val_loader:

                # show images and labels
                # self.show_images(images, labels)

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.network(images)
                _, predicted = outputs.max(1)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

            print(f"validation part({self.current_epoch - 1}ep):")
            print("predicted_labels, true_labels")
            print(predicted_labels, true_labels)
            print()

        self.val_metrics = self.calculate_metrics(true_labels, predicted_labels) + (val_loss,)

    def test(self):
        self.network.eval()
        test_loss = 0.0
        predicted_labels = []
        true_labels = []

        if self.test_with_best_val_loss:
            self.network.load_state_dict(self.best_model_state_dict)
            self.run.summary["best_val_loss_epoch"] = self.epochs_with_best_val_loss
            print(f"Test with best val loss model from {self.epochs_with_best_val_loss} epoch, totally {self.current_epoch - 1} epochs")

        else:
            self.run.summary["best_val_loss_is_last"] = self.current_epoch - 1
            print(f"Test with last model from {self.current_epoch - 1} epoch")


        with no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.network(images)
                _, predicted = outputs.max(1)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

            print(f"test part({self.current_epoch - 1}ep):")
            print("predicted_labels, true_labels")
            print(predicted_labels, true_labels)
            print()

        self.test_metrics = self.calculate_metrics(true_labels, predicted_labels) + (test_loss,)

    def send_to_wandb(self, is_test=False):

        if is_test:
            test_confusion_matrix = Table(columns=["TN", "FP", "FN", "TP"])
            test_confusion_matrix.add_data(*(self.test_metrics[4].ravel()))

            test_log = {
                "test_loss": self.test_metrics[5],
                "test_precision": self.test_metrics[0],
                "test_recall": self.test_metrics[1],
                "test_f1": self.test_metrics[2],
                "test_balanced_acc": self.test_metrics[3],
                "test_confusion_matrix": copy(test_confusion_matrix)
            }

            self.run.log(test_log)
            return

        val_log, train_log = {}, {}

        if self.val_loader:
            self.val_confusion_matrix.add_data(*(self.val_metrics[4].ravel()))

            val_log = {
                "val_loss": self.val_metrics[5],
                "val_precision": self.val_metrics[0],
                "val_recall": self.val_metrics[1],
                "val_f1": self.val_metrics[2],
                "val_balanced_acc": self.val_metrics[3],
                "val_confusion_matrix": copy(self.val_confusion_matrix)
            }

        self.train_confusion_matrix.add_data(*(self.train_metrics[4].ravel()))

        run_log = {
            "train_loss": self.train_metrics[5],
            "train_precision": self.train_metrics[0],
            "train_recall": self.train_metrics[1],
            "train_f1": self.train_metrics[2],
            "train_balanced_acc": self.train_metrics[3],
            "train_confusion_matrix": copy(self.train_confusion_matrix)
        }


        print("logs")
        print(val_log, type(val_log))
        print(run_log, type(run_log))



        self.run.log({**val_log, **run_log})

    def calculate_metrics(self, true_labels, predicted_labels):
        precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
        recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
        f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        confusion_matrix_output = confusion_matrix(true_labels, predicted_labels)

        return precision, recall, f1, balanced_accuracy, confusion_matrix_output


    def save_model(self):
        date = datetime.now().strftime("%Y-%m-%d")
        save(self.best_model_state_dict,
             self.run.config.weights_dir +
             f"{date}_epoch:{self.current_epoch}_run-id:{self.run.id}_val-loss:{self.val_metrics[5]}_{self.run.config.fold}.pth")

    def is_stopping(self):

        if self.early_stopping:
            if self.val_metrics[5] < self.best_val_loss:
                self.best_val_loss = self.val_metrics[5]
                self.best_model_state_dict = self.network.state_dict()
                self.epochs_without_improvement = 0
                self.epochs_with_best_val_loss = self.current_epoch - 1

                if self.save_weights and not self.run.config.saving_only_best_one_weight_val_loss:
                    self.save_model()

            else:

                # TODO: Add here ability to fine-tining
                # if fine_tune_enable:

                self.epochs_without_improvement += 1
                print(f"Epochs without improvement: {self.epochs_without_improvement}/ {self.patience}")

                if self.epochs_without_improvement >= self.patience:
                    if self.run:
                        self.run.summary["stopping"] = self.current_epoch

                    print(f"Early stopping after {self.current_epoch} epochs")

                    if self.run.config.saving_only_best_one_weight_val_loss:
                        self.save_model()

                    return True

        return False
