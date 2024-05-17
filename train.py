import torch
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from agent import *
from models import *
from data import *


import os


def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html   Stand 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def train(model_name, n_classes, max_epochs, base_model=None, reinforce=True):
    #torch.autograd.set_detect_anomaly(True)
    if not os.path.exists("./saves"):
        os.makedirs("./saves/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    train_loader, test_loader = get_loader()
    if reinforce:
        print("Reinforce")
        agent = Agent(196)
        agent = torch.nn.DataParallel(agent)
        agent = agent.to(device)
        agent_optimizer = optim.RMSprop(agent.parameters(), lr=0.00025)
    else:
        agent = None
    if base_model:
        model = ViT(n_classes, device=device, pretrained=True, reinforce=reinforce)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(base_model), strict=False)

        model = model.to(device)
        # print("Running base evaluation")
        # class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes)
        # print('[Test] ACC: {:.4f} '.format(accuracy))
        # print(f'[Test] CLASS ACC: {class_accuracy} @{-1}')
    else:
        model = ViT(n_classes, device=device, pretrained=True, reinforce=reinforce)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)
    scheduler = optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.9)
    scaler = GradScaler()

    best_acc = 0
    train_agent = True
    for epoch in range(max_epochs):
        if reinforce:
            loss, acc = train_rl(train_loader, device, model, model_optimizer, scaler, agent, train_agent=False)
            loss, acc = train_rl(train_loader, device, model, agent_optimizer, scaler, agent, train_agent=True)



        else:
            loss, acc = train_vit(train_loader, device, model, model_optimizer, scaler)
        class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes, agent)
        print('[Test] ACC: {:.4f} '.format(accuracy))
        print(f'[Test] CLASS ACC: {class_accuracy} @{epoch}')
        summarize(writer, "test", epoch, accuracy)
        if accuracy > best_acc:
            best_acc = accuracy

            torch.save(model.state_dict(), f"saves/best_{model_name}_@{launch_time}.pth")
            torch.save(agent.state_dict(), f"saves/best_{model_name}_{agent}_@{launch_time}.pth")
        summarize(writer,"train", epoch, acc, loss)

        scheduler.step(epoch)


def summarize(writer, split, epoch, acc, loss=None):
    writer.add_scalar('accuracy/'+ split, acc, epoch)
    if loss:
        writer.add_scalar('CE_Loss/' + split, loss, epoch)

def eval_vit(model, device, loader, n_classes, agent):
    model.eval()
    correct = torch.zeros(n_classes)
    overall = torch.zeros(n_classes)
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            if agent is not None:
                q_table, values = agent(inputs)
                prob = torch.exp(q_table)
                dist = Categorical(prob)
                action = dist.sample()
                outputs = model(inputs, action.detach())
            else:
                outputs = model(inputs, None)
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


            outputs = model(inputs, None)
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

def train_rl(loader, device, model, optimizer, scaler, agent, train_agent):
    criterion = torch.nn.CrossEntropyLoss()

    if train_agent:
        model.eval()
        agent.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in agent.parameters():
            param.requires_grad = True
    else:
        model.train()
        agent.eval()
        for param in model.parameters():
            param.requires_grad = True
        for param in agent.parameters():
            param.requires_grad = False
        for param in agent.module.backbone.parameters():
            param.requires_grad = False
    correct = 0
    n_items = 0
    running_loss = 0
    best_acc = 0
    unchanged = 0
    counter = 0
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            q_table, values = agent(inputs)
            prob = torch.exp(q_table)
            dist = Categorical(prob)
            action = dist.sample()
            outputs = model(inputs, action.detach())
            probs, preds = torch.max(outputs, 1)
            rewards = (preds == labels).long()
            if train_agent:
                log_prob = dist.log_prob(action)
                values = get_values(rewards, dist.log_prob(action))
                loss = torch.mean((-log_prob) * values)

            else:

                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct += torch.sum(preds == labels)
        n_items += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

        if counter % 100 == 99:
            print(f'Reinforce_Loss {loss}')
            acc = correct / n_items
            print(acc)
            if acc > best_acc+0.001:
                best_acc = acc
                unchanged = 0
            else:
                unchanged += 1
                if unchanged > 10:
                    break
        counter += 1

    return running_loss, correct / n_items


def get_values(reward, log_probs):
    gamma = 0.9
    val = torch.zeros_like(log_probs)
    val[:, -1] = reward
    val[val == 0] = -0.01
    batch_size, seq_len = val.shape
    # Create a discount factors tensor [1, gamma, gamma^2, ..., gamma^(seq_len-1)]
    discount_factors = gamma ** torch.arange(seq_len, dtype=torch.float32, device=val.device)

    # Calculate discounted rewards by multiplying rewards with discount_factors
    discounted_rewards = val * discount_factors.unsqueeze(0)

    # Compute cumulative sums in reverse to get the Q-values
    q_values = torch.flip(torch.cumsum(torch.flip(discounted_rewards, dims=[1]), dim=1), dims=[1])

    # Normalize by dividing by the discount_factors to undo the initial scaling
    q_values = q_values / discount_factors.unsqueeze(0)

    return q_values


if __name__ == "__main__":
    set_deterministic()
    num_classes = 10
    max_epochs = 300
    base = "saves/baseline.pth"
    model = "rl_learning"
    train(model, num_classes, max_epochs, base, reinforce=True)