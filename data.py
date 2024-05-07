from typing import Literal, Sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

import torch
import numpy as np
class Data(Dataset):
    def __init__(self, size, split=None):

        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.split = split

        self.resize = transforms.Resize(size)

        if split =="train":
            self.flip = transforms.RandomHorizontalFlip(0.1)

            self.train_transform = [(transforms.RandomCrop(int(size * 0.9)), 0.2), (transforms.RandomHorizontalFlip(1), 0.1),
                                    (transforms.ColorJitter(
                                        brightness=0.5, contrast=1, saturation=0.1, hue=0.5), 0.2),
                                    (transforms.RandomRotation(
                                        degrees=20
                                    ),  0.2),
                                    (transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), 0.1),
                                    ]
            self.set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True)
        else:
            self.set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True)


    def __len__(self):
        return len(self.set)
    def __getitem__(self, idx):
        image, label = self.set[idx]
        image = self.resize(image)
        if self.split == "train":
            choices = np.random.random_sample(size=len(self.train_transform))
            for i, t in enumerate(self.train_transform):
                trans, prob = t
                if choices[i] < prob:
                    image = trans(image)
        if self.transform:
            image = self.transform(image)

        return image, label

def get_loader():

    train_data = Data(size=64, split="train")
    test_data = Data(size=64, split="test")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8,
                                             shuffle=False, num_workers=4)

    return train_loader, test_loader


