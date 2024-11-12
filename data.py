from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import random
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
class Data(Dataset):
    def __init__(self, size, split=None, imagenet=True, dataset='calltech', data=None):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if imagenet else transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform = transforms.Compose(
            [
                normalize
             ])
        self.split = split
        self.size = size

        self.resize = transforms.Resize((size, size))

        if split =="train":

            self.train_transform = [(transforms.RandomCrop(32, padding=4), 1), (transforms.RandomHorizontalFlip(1), 0.5),
                                    (transforms.RandomVerticalFlip(1), 0.5),

                                    (transforms.RandomRotation(
                                        degrees=30
                                    ),  0.5),
                                    (transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), 0.5)
                                    ]
            if data is not None:
                self.set = data
            else:
                if dataset == 'cifar10':
                    self.set = torchvision.datasets.CIFAR10(root='./data',  train=True,
                                                               download=True)
                else:
                    self.set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                            download=True)


        else:
            if data is not None:
                self.set = data
            else:
                if dataset == 'cifar10':
                    self.set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                            download=True)
                else:
                    self.set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                             download=True)

    def __len__(self):
        return len(self.set)
    def __getitem__(self, idx):
        image, label = self.set[idx]
        image = transforms.Resize((self.size, self.size))(image)
        image = transforms.ToTensor()(image)
        #image = self.resize(image)
        # if self.split == "train":
        #     choices = np.random.random_sample(size=len(self.train_transform))
        #     for i, t in enumerate(self.train_transform):
        #         trans, prob = t
        #         if choices[i] <= prob:
        #             image = trans(image)

        if image.shape[0] == 1:

            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_loader(size, bs, world_size=1, rank=0, dataset="caltech101"):
    train_set = val_set = None
    if dataset =="caltech101":
        data = torchvision.datasets.Caltech101(root='./data', download=True)
        length = len(data)
        train_len = int(length * 0.7)
        train_set, val_set = torch.utils.data.random_split(data, [train_len, length-train_len])
    train_data = Data(size=size, split="train", data=train_set, dataset=dataset)
    test_data = Data(size=size, split="test", data=val_set, dataset=dataset)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, num_workers=6, sampler=train_sampler, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, num_workers=6, sampler=test_sampler, pin_memory=False)

    return train_loader, test_loader


def display_images(loader):
    margin = 50  # pixels
    spacing = 35  # pixels
    dpi = 100.  # dots per inch
    width = (400 + 200 + 2 * margin + spacing) / dpi  # inches
    height = (180 + 180 + 2 * margin + spacing) / dpi
    left = margin / dpi / width  # axes ratio
    bottom = margin / dpi / height
    wspace = spacing / float(200)
    fig, axes = plt.subplots(3, 6, figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom,
                        wspace=wspace, hspace=wspace)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    for imgs, labels in loader:
        imgs = unnormalize(imgs)
        for ax, img, label in zip(axes.flatten(), imgs, labels):
            ax.axis('off')
            ax.imshow(img.permute(1, 2, 0))

        plt.show()
        break

def patchify_image(loader):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    patch_width = int(224 / 4)
    unfold = torch.nn.Unfold(kernel_size=patch_width, stride=patch_width)
    for imgs, labels in loader:
        print(imgs.shape)
        bs, c, H, W = imgs.shape


        n_rows = H // patch_width
        n_cols = W // patch_width


        x = imgs
        x = unfold(x)
        # x -> B (c*p*p) L

        # Reshaping into the shape we want
        a = x.view(bs, c, patch_width, patch_width, -1).permute(0, 4, 1, 2, 3)
        a = unnormalize(a)
        f, axs = plt.subplots(n_rows, n_cols, figsize=(5, 5))

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                axs[row_idx, col_idx].imshow(a[0, col_idx + 4 * row_idx, ...].permute(1, 2, 0))

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        f.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        f, axs = plt.subplots(n_rows * n_cols, 1, figsize=(1, 5))

        for idx in range(n_rows * n_cols):
                axs[idx].imshow(a[0, idx, ...].permute(1, 2, 0))

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        f.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

        random_perm = random.sample(range(n_rows * n_cols), n_rows * n_cols)
        f, axs = plt.subplots(n_rows * n_cols, 1, figsize=(1, 5))

        for e, idx in enumerate(random_perm):
            axs[e].imshow(a[0, idx, ...].permute(1, 2, 0))

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        f.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        break


if __name__ == "__main__":
    data = torchvision.datasets.Caltech101(root='./data', download=True)
    train_data = Data(size=224, split="train", data=data)
    train_loader = torch.utils.data.DataLoader(train_data, 2, shuffle=True)
    patchify_image(train_loader)
    #display_images(train_loader)