from datetime import datetime

from torch import no_grad, save
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from wandb import Table
from torch import device as torch_device


class Trainer:
    def __init__(self, network, train_loader, val_loader, test_loader, optimizer, criterion, run, device_name, patience=-1,
                 early_stopping: bool = True or False, save_weights: bool = True or False):

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

        self.train_confusion_matrix = Table(columns=["TP", "FP", "TN", "FN"])
        self.val_confusion_matrix = Table(columns=["TP", "FP", "TN", "FN"])
        self.test_confusion_matrix = Table(columns=["TP", "FP", "TN", "FN"])

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state_dict = None
        self.save_weights = save_weights

        self.current_epoch = 0

    def train_epoch(self):
        self.network.train()
        train_loss = 0.0
        predicted_labels = []
        true_labels = []

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.network(images)
            _, predicted = outputs.max(1)
            loss = self.criterion(outputs, labels)
            train_loss += loss
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            loss.backward()
            self.optimizer.step()

        self.train_metrics = self.calculate_metrics(true_labels, predicted_labels) + (train_loss,)

        self.current_epoch += 1

    def validate(self):
        self.network.eval()
        val_loss = 0.0
        predicted_labels = []
        true_labels = []

        with no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.network(images)
                _, predicted = outputs.max(1)
                loss = self.criterion(outputs, labels)
                val_loss += loss
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        self.val_metrics = self.calculate_metrics(true_labels, predicted_labels) + (val_loss,)

    def test(self):
        self.network.eval()
        test_loss = 0.0
        predicted_labels = []
        true_labels = []

        with no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.network(images)
                _, predicted = outputs.max(1)
                loss = self.criterion(outputs, labels)
                test_loss += loss
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        self.test_metrics = self.calculate_metrics(true_labels, predicted_labels) + (test_loss,)

    def send_to_wandb(self, is_test=False):

        if is_test:
            self.test_confusion_matrix.add_data(*(self.test_metrics[4].ravel()))

            test_log = {
                "test_loss": self.test_metrics[5],
                "test_precision": self.test_metrics[0],
                "test_recall": self.test_metrics[1],
                "test_f1": self.test_metrics[2],
                "test_balanced_acc": self.test_metrics[3],
                "test_confusion_matrix": self.test_confusion_matrix
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
                "val_confusion_matrix": self.val_confusion_matrix
            }

        self.train_confusion_matrix.add_data(*(self.train_metrics[4].ravel()))

        run_log = {
            "train_loss": self.train_metrics[5],
            "train_precision": self.train_metrics[0],
            "train_recall": self.train_metrics[1],
            "train_f1": self.train_metrics[2],
            "train_balanced_acc": self.train_metrics[3],
            "train_confusion_matrix": self.train_confusion_matrix
        }

        self.run.log({**val_log, **run_log})

    def calculate_metrics(self, true_labels, predicted_labels):
        precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
        recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
        f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        confusion_matrix_output = confusion_matrix(true_labels, predicted_labels)

        return precision, recall, f1, balanced_accuracy, confusion_matrix_output

    def is_stopping(self):

        if self.early_stopping:
            if self.val_metrics[5] < self.best_val_loss:
                self.best_val_loss = self.val_metrics[5]
                self.best_model_state_dict = self.network.state_dict()
                self.epochs_without_improvement = 0

                if self.save_weights:
                    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save(self.best_model_state_dict, f"weights/{date}_run-id:{self.run.id}_val-loss:{self.val_metrics[5]}.pth")

            else:

                # TODO: Add here ability to fine-tining
                # if fine_tune_enable:

                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    if self.run:
                        self.run.summary["stopping"] = self.current_epoch

                    print(f"Early stopping after {self.current_epoch} epochs")
                    return True

        return False