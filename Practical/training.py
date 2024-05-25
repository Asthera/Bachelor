from os.path import join
from time import time
import wandb
from transforms.builder import TransformsBuilder
from utils.trainer import Trainer
import yaml
from utils.build import build_datasets, build_network, build_criterion, build_optimizer
import torch
from torch.utils.data import DataLoader
import sys
import random
import numpy as np


def find_yaml_file_arg(argv):
    """Parse command line arguments for a --yaml_file argument."""
    for arg in argv:
        if arg.startswith('--yaml_file='):
            return arg.split('=', 1)[1]
    return None


def train(config_path: str):
    # set deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

    init_time = time()

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # giving a name to the run
    wandb.run.name = f"tr:{run.config.transform}_{run.config.fold}_batch:{run.config.batch_size}_{run.id}"

    # for example = wandb.config.fold = "fold_0"
    fold_json_path = str(join(run.config.folds_path, run.config.fold + ".json"))

    # TRANSFORMS
    init_transform = TransformsBuilder(run.config.init_transform).build()
    train_target_transform = TransformsBuilder(run.config.transform).build()
    output_transform = TransformsBuilder(run.config.output_transform).build()

    print("Transforms:")
    print(run.config.init_transform, run.config.transform, run.config.output_transform)
    print(init_transform, train_target_transform, output_transform)
    print()

    # DATASETS
    train_dataset, val_dataset, test_dataset = build_datasets(fold_json_path,
                                                              run.config.img_dir,
                                                              init_transform,
                                                              train_target_transform,
                                                              output_transform,
                                                              run.config.experiment_class,
                                                              is_validating=run.config.is_validating,
                                                              val_dist=run.config.val_dist)

    # DATALOADERS
    train_loader, test_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True), \
        DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=run.config.batch_size, shuffle=False)
    else:
        val_loader = None

    # NETWORK
    network = build_network(arch=run.config.arch,
                            fc_layer_size=run.config.fc_layer_size,
                            number_of_classes=run.config.number_of_classes,
                            pretrained=run.config.pretrained,
                            device_name=run.config.device,
                            dropout=run.config.dropout,
                            dropout_rate=run.config.dropout_rate)

    criterion = build_criterion(criterion_name=run.config.criterion)
    optimizer = build_optimizer(network, run.config.optimizer, run.config.learning_rate)

    # Train model
    trainer = Trainer(network, train_loader, val_loader, test_loader, optimizer, criterion, run,
                      device_name=run.config.device, patience=run.config.patience,
                      early_stopping=run.config.early_stopping, save_weights=run.config.save_weights,
                      test_with_best_val_loss=run.config.testing_with_best_val_loss)

    print(f"Initialization took {time() - init_time} seconds")

    for epoch in range(run.config.epochs):

        start_time = time()
        trainer.train_epoch()

        if val_loader is not None:
            trainer.validate()

        trainer.send_to_wandb()

        if trainer.is_stopping():
            break

        print(f"Train epoch took {time() - start_time} seconds")

    # Test model
    trainer.test()
    trainer.send_to_wandb(is_test=True)

    wandb.finish()

    print(f"Total time: {time() - init_time} seconds")


yaml_file = find_yaml_file_arg(sys.argv)
train(yaml_file)