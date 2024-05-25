from torch.nn import CrossEntropyLoss
from torch.nn import Linear, Sequential, Dropout
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from torch import device as torch_device
from torch.optim import Adam, SGD
import sys

sys.path.append("..")
from data.dataset import FramesDataset


def build_criterion(criterion_name):
    if criterion_name == "cross-entropy":
        return CrossEntropyLoss()
    else:
        raise ValueError("Invalid criterion")


def build_network(arch: str, fc_layer_size: int, number_of_classes: int, pretrained: bool, device_name: str,
                  dropout: bool, dropout_rate: float):
    if arch != "resnet18":
        raise ValueError("Invalid architecture")

    device = torch_device(device_name)

    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    model.fc = Linear(fc_layer_size, number_of_classes)

    if dropout:
        model.fc = Sequential(
            Dropout(dropout_rate, inplace=True),
            Linear(fc_layer_size, number_of_classes)
        )

    return model.to(device)


def build_optimizer(network, optimizer_name, learning_rate):
    if optimizer_name == "adam":
        return Adam(network.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return SGD(network.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")


def build_datasets(json_path: str,
                   img_dir: str,
                   init_transform,
                   train_target_transform,
                   output_transform,
                   experiment_class: str = "b_line" or "a_line",
                   is_validating: bool = False,
                   val_dist: float = 0.2):
    # DATASETS

    if is_validating:
        val_dataset = FramesDataset(json_path, img_dir, init_transform, None, output_transform, subset="val",
                                    experiment_class=experiment_class, val=val_dist)
        train_dataset = FramesDataset(json_path, img_dir, init_transform, train_target_transform, output_transform, subset="train",
                                      experiment_class=experiment_class, val=val_dist)
    else:
        val_dataset = None
        train_dataset = FramesDataset(json_path, img_dir, init_transform, train_target_transform, output_transform,
                                      subset="train", experiment_class=experiment_class)

    test_dataset = FramesDataset(json_path, img_dir, init_transform, None, output_transform, subset="test",
                                 experiment_class=experiment_class)

    return train_dataset, val_dataset, test_dataset
