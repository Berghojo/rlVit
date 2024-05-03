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


def train(model_name, n_classes, max_epochs, base_model=None):
    #torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = ViT(n_classes, device = device)
    if base_model:
        model.load_state_dict(torch.load(base_model))
    model = torch.nn.DataParallel(model)
    train_loader, test_loader = get_loader()
    optimizer = optim.Adam(model.parameters(), lr=8e-4, weight_decay=0.1)
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)

    scaler = GradScaler()
    model.to(device)
    best_acc = 0
    for epoch in range(max_epochs):

        loss, acc = train_vit(train_loader, device, model, optimizer, scaler)
        class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes)
        print('[Test] ACC: {:.4f}'.format(accuracy))
        print(f'[Test] CLASS ACC: {class_accuracy}')
        summarize(writer, "test", epoch, accuracy)
        if accuracy > best_acc:
            torch.save(model.state_dict(), f"saves/best_{model_name}_@{launch_time}.pth")
        summarize(writer,"train", epoch, acc, loss)

    raise NotImplementedError

def summarize(writer, split, epoch, acc, loss=None):
    writer.add_scalar('accuracy/'+ split, acc, epoch)
    if loss:
        writer.add_scalar('CE_Loss/' + split, loss, epoch)

def eval_vit(model, device, loader, n_classes):
    model.eval()
    correct = torch.zeros(n_classes)
    overall = torch.zeros(n_classes)
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i, boolean in enumerate(preds == labels):
                overall[preds[i]] += 1
                if boolean:
                    correct[preds[i]] += 1
    class_accuracy = torch.tensor(correct) / torch.tensor(overall)
    accuracy = sum(correct) / sum(overall)
    print(correct)
    print(overall)
    return class_accuracy, accuracy
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
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct += torch.sum(preds == labels)
        n_items += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

        if counter % 1000 == 999:
            print(correct / n_items)
        counter += 1

    return running_loss, correct / n_items


if __name__ == "__main__":
    set_deterministic()
    num_classes = 10
    max_epochs = 300
    base = "saves/hr.pth"
    base = None
    model = "base"
    train(model, num_classes, max_epochs, base)
