from typing import Literal, Sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch

class Data(Dataset):
    def __init__(self, size):
        transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

def get_loader():
    data = Data(size=224)
    train_loader = torch.utils.data.DataLoader(data.train_set, batch_size=4,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(data.test_set, batch_size=4,
                                             shuffle=False, num_workers=2)
    return train_loader, test_loader


