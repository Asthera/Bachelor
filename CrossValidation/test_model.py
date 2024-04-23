# in this file we will test the model using the test data
# you just need to pass a path to model weights and the test data

import sys
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from transforms.builder import TransformsBuilder
from torch.nn import Linear
from torch import load, no_grad
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

sys.path.append("..")
from data.dataset import FramesDataset


def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    confusion_matrix_output = confusion_matrix(true_labels, predicted_labels)
    confusion_matrix_output = confusion_matrix_output.ravel()
    return (precision, recall, f1, balanced_accuracy, confusion_matrix_output)


def test_model(model_path, test_data_path, test_data_json):
    # we are using pretrained resnet-18 model
    network = resnet18(weights=ResNet18_Weights.DEFAULT)
    network.fc = Linear(512, 2)

    # load the model weights
    network.load_state_dict(load(model_path))

    # load the test data by test_loader

    init_transform = TransformsBuilder(["transform_resize(600)", "transform_pad(600)"]).build()

    test_dataset = FramesDataset(test_data_json, test_data_path, init_transform, target_transform=None,
                                 output_transform=None, subset="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # test the model
    network.eval()
    predicted_labels = []
    true_labels = []

    with no_grad():
        for images, labels in test_loader:
            outputs = network(images)
            _, predicted = outputs.max(1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("predicted_labels, true_labels")
    print(predicted_labels, true_labels)
    print()

    # calculate the metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    print("Metrics:")
    metrics = ["Precision: " + str(metrics[0]), "Recall: " + str(metrics[1]), "F1: " + str(metrics[2]),
               "Balanced Accuracy: " + str(metrics[3]), "Confusion Matrix: " + str(metrics[4])]
    print(metrics)


model_path = "weights/2024-03-19_14-40-10_epoch:6_run-id:jhb95v3s_val-loss:0.1636328101158142.pth"
test_data_path = "/Volumes/LaCie/Macbook/Bachelor/CrossValidation/"
test_data_json = "metadata_folds/kkui-lung-bline-lumify:latest/video/fold_0.json"

# ["TN", "FP", "FN", "TP"] = [112  91  24   3] for confusion matrix
test_model(model_path, test_data_path, test_data_json)
