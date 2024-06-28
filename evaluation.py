# -*- coding: utf-8 -*-
import torch.utils.data





def evaluation(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    raise NotImplementedError


def load_and_evaluate_model(model_name: str, weights_path: str, dataset_path: str, result_file: str):
    """
    Loads a model and evaluates it on the test set, writes the file for the leaderboard
    :param model_name: resnet or hrnet
    :param weights_path: path to saved weights
    :param dataset_path: path to the dataset, general location, NOT specifically the test set
    :param result_file: path to the file that is written for the leaderboard, contains the results
    :return: nothing
    """

    raise NotImplementedError
