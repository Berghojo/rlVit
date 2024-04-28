# -*- coding: utf-8 -*-
from models import ViT
from data import get_loader
import torch
import torchvision
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
from tqdm import tqdm
import os


def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html   Stand 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def train(model_name, n_classes, max_epochs):
    if model_name == "base":
        model = ViT(n_classes, pretrained=True)
    train_loader, test_loader = get_loader()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    model.to(device)
    for epoch in range(max_epochs):
        loss, acc = train_vit(train_loader, device, model, optimizer, scaler)
        summarize(writer,"train", epoch, acc, loss)

    raise NotImplementedError

def summarize(writer, split, epoch, acc, loss=None):
    writer.add_scalar('accuracy/'+ split, acc, epoch)
    if loss:
        writer.add_scalar('CE_Loss/' + split, loss, epoch)

def train_vit(loader, device, model, optimizer, scaler):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    correct = 0
    n_items = 0
    running_loss = 0
    counter = 0
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)
        n_items += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

        if counter % 100 == 0:
            print(correct / n_items)
            correct = 0
            n_items = 0
        counter += 1

    return running_loss, correct / n_items


if __name__ == "__main__":
    set_deterministic()
    num_classes = 10
    max_epochs = 100
    model = "base"
    train(model, num_classes, max_epochs)
