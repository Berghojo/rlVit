from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
class Data(Dataset):
    def __init__(self, size, split=None, imagenet=False):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if imagenet else transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize
             ])
        self.split = split

        self.resize = transforms.Resize(size)

        if split =="train":

            self.train_transform = [(transforms.RandomCrop(32, padding=4), 1), (transforms.RandomHorizontalFlip(1), 0.5),
                                    (transforms.RandomVerticalFlip(1), 0.5),
                                    (transforms.RandomRotation(
                                        degrees=30
                                    ),  0.5),
                                    (transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), 0.5)
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

        #image = self.resize(image)
        # if self.split == "train":
        #     choices = np.random.random_sample(size=len(self.train_transform))
        #     for i, t in enumerate(self.train_transform):
        #         trans, prob = t
        #         if choices[i] <= prob:
        #             image = trans(image)
        if self.transform:
            image = self.transform(image)

        return image, label

def get_loader(size, bs, world_size=1, rank=0):

    train_data = Data(size=size, split="train")
    test_data = Data(size=size, split="test")
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs,
                                               shuffle=False, num_workers=6, sampler=train_sampler, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs,
                                             shuffle=False, num_workers=6, sampler=test_sampler, pin_memory=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_data = Data(size=224, split="train")
    train_loader = torch.utils.data.DataLoader(train_data, 10, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for imgs,labels in train_loader:
        print(labels.shape, imgs.shape)
        for img, label in zip(imgs, labels):
            plt.imshow(img.permute(1, 2, 0))
            plt.xlabel(f"{classes[label]}")
            plt.show()
        break