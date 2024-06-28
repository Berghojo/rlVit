# -*- coding: utf-8 -*-

import importlib
import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import sys

import torch


def cli():
    parser = ArgumentParser()
    parser.add_argument(
            "input", help="Input archive (zip file) of your submission that should be checked for testability"
            )
    args = parser.parse_args()

    inpath = Path(args.input)

    assert inpath.is_file() and inpath.suffix == ".zip", "Input must be a zip file"
    with TemporaryDirectory(prefix="testability_extract_") as unpackdir:
        with ZipFile(inpath, "r") as submissions_archive:
            submissions_archive.extractall(unpackdir)
            filename = os.path.basename(args.input)[:-4]  # filename without path and without .zip extension

            sys.path.insert(1, f'{unpackdir}')
            module = importlib.import_module(f"{filename}.heatmaps")
            joints = [[1, 1, 1],
                      [1, 1, 0],
                      [0, 0, 0]]
            test_map = module.create_heatmaps(joints)
            assert test_map.shape[0] == 3

            module = importlib.import_module(f"{filename}.dataset")
            func1 = module.load_dataset
            func2 = module.create_skijump_subsets
            class1 = module.SkijumpDataset
            func3 = class1.augment

            module = importlib.import_module(f"{filename}.metric")
            func4 = module.pck

            module = importlib.import_module(f"{filename}.models")
            model1 = module.ResNet18Model
            assert issubclass(model1, torch.nn.Module)
            model2 = module.HRNetModel
            assert issubclass(model2, torch.nn.Module)

            module = importlib.import_module(f"{filename}.train")
            func5 = module.train
            func6 = module.train_resnet18_model
            func7 = module.train_hrnet_model

            module = importlib.import_module(f"{filename}.evaluation")
            func8 = module.get_coordinates_from_heatmaps
            func9 = module.evaluation

    print("Submission for Assignment 6 successfully checked. You can now upload it to Digicampus.")


if __name__ == "__main__":
    cli()
