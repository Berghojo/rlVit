import torch
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
import gc
from tqdm import tqdm
from torch.distributions.categorical import Categorical

import os

from util import CustomLoss
from agent import *
from models import *
from data import *

def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html   Stand 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def train(model_name, n_classes, max_epochs, base_model=None, reinforce=True, pretrained=True, agent_model=None,
          verbose=True, img_size=224, base_vit=False, batch_size=32):
    # torch.autograd.set_detect_anomaly(True)

    if not os.path.exists("./saves"):
        os.makedirs("./saves/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)
    train_loader, test_loader = get_loader(img_size, batch_size)
    if reinforce:
        print("Reinforce")
        agent = SimpleAgent(196)
        agent = torch.nn.DataParallel(agent)
        agent = agent.to(device)
        agent_optimizer = optim.Adam(agent.parameters(), lr=1e-4)
        if agent_model is not None:
            agent.load_state_dict(torch.load(agent_model))
    else:
        agent = None
    if base_model:
        model = ViT(n_classes, device=device, pretrained=pretrained, reinforce=reinforce)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(base_model), strict=False)

        model = model.to(device)
        class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes, agent, verbose=verbose)
        print('[Test] ACC: {:.4f} '.format(accuracy))
        print(f'[Test] CLASS ACC: {class_accuracy} @-1')
        summarize(writer, "test", -1, accuracy)
    else:
        model = ViT(n_classes, device=device, pretrained=pretrained, reinforce=reinforce) if not base_vit else BaseVit(10, pretrained)
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, max_epochs)
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(max_epochs):
        if reinforce:
            loss, acc, = train_rl(train_loader, device, model, model_optimizer, scaler, agent, train_agent=False,
                                  verbose=verbose)

            agent_loss, agent_acc, policy_loss, entropy_loss = train_rl(train_loader, device, model,
                                                                        agent_optimizer, scaler, agent,
                                                                        train_agent=True, verbose=verbose)





            summarize(writer, "train_agent", epoch, agent_acc, agent_loss)
            summarize_agent(writer, "train_agent", epoch, policy_loss, entropy_loss)


        else:
            loss, acc = train_vit(train_loader, device, model, model_optimizer, scaler, verbose=verbose)
        summarize(writer, "train", epoch, acc, loss)
        class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes, agent, verbose=verbose)
        print('[Test] ACC: {:.4f} '.format(accuracy))
        print(f'[Test] CLASS ACC: {class_accuracy} @{epoch}')
        summarize(writer, "test", epoch, accuracy)
        if accuracy > best_acc:
            best_acc = accuracy

            torch.save(model.state_dict(), f"saves/best_{model_name}_@{launch_time}.pth")
            if agent is not None:
                torch.save(agent.state_dict(), f"saves/best_{model_name}_agent_@{launch_time}.pth")

        scheduler.step()


def summarize(writer, split, epoch, acc, loss=None):

    writer.add_scalar('accuracy/' + split, acc, epoch)
    if loss:
        writer.add_scalar('Loss/' + split, loss, epoch)
def summarize_agent(writer, split, epoch, policy_loss, entropy_loss):
    writer.add_scalar('policy_loss/' + split, policy_loss, epoch)
    writer.add_scalar('entropy_loss/' + split, entropy_loss, epoch)
def eval_vit(model, device, loader, n_classes, agent, verbose=True):
    model.eval()
    agent.eval()
    correct = torch.zeros(n_classes)
    overall = torch.zeros(n_classes)
    with torch.no_grad():
        for inputs, labels in tqdm(loader, disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            if agent is not None:
                bs, _, _, _ = inputs.shape
                start = torch.full((bs, 196), 196, dtype=torch.long, device=device)
                for i in range(196):
                    state = model.module.get_state(inputs, start)
                    actions, values = agent(state)

                    action = torch.argmax(actions, dim=-1)
                    start[:, i] = action

                outputs = model(inputs, start)
            else:
                outputs = model(inputs, None)
            _, preds = torch.max(outputs, 1)

            for i, boolean in enumerate(preds == labels):
                overall[preds[i]] += 1
                if boolean:
                    correct[preds[i]] += 1
    if agent is not None:

        test_input, _ = next(iter(loader))
        test_input = torch.unsqueeze(test_input[0], 0)
        q_table, values = agent(test_input)
        f = open("permutation.txt", "a")
        print(q_table.shape)
        values, action = torch.max(q_table, dim=-1)
        print(list(action), file=f)
        print(list(values), file=f)
    class_accuracy = torch.tensor(correct) / torch.tensor(overall)
    accuracy = sum(correct) / sum(overall)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(classes)
    print(correct)
    print(overall)
    return class_accuracy, accuracy


def train_vit(loader, device, model, optimizer, scaler, verbose=True):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    correct = 0
    n_items = 0
    running_loss = 0
    counter = 0
    for inputs, labels in tqdm(loader, disable=not verbose):
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
        del outputs
        if counter % 1000 == 999:
            print(correct / n_items)
        counter += 1

    return running_loss, correct / n_items


def train_rl(loader, device, model, optimizer, scaler, agent, train_agent, verbose=True, rl=True):
    criterion = torch.nn.CrossEntropyLoss()

    if train_agent:

        criterion2 = torch.nn.CrossEntropyLoss(ignore_index=196, label_smoothing=0.7)
        loss_func = CustomLoss().to(device)
        model.eval()
        agent.train()
        agent.module.freeze(train_agent)
        model.module.freeze(not train_agent)
    else:
        model.train()
        agent.eval()
        agent.module.freeze(train_agent)
        model.module.freeze(not train_agent)
    correct = 0
    n_items = 0
    running_loss = 0

    counter = 0
    for inputs, labels in tqdm(loader, disable=not verbose):
        inputs = inputs.to(device)
        bs, _, _, _ = inputs.shape
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        start = torch.full((bs, 196), 196, dtype=torch.long, device=device)
        val = torch.zeros((bs, 196), dtype=torch.float32, device=device)
        action_probs = torch.zeros((bs, 196), dtype=torch.float32, device=device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if train_agent:

                for i in range(196):
                    state = model.module.get_state(inputs, start)
                    actions, values = agent(state)

                    prob = torch.exp(actions)
                    dist = Categorical(prob)
                    action = dist.sample()

                    action_prob = prob.gather(-1, action.unsqueeze(-1))
                    action_probs[:, i] = action_prob.squeeze()
                    start[:, i] = action
                    val[:, i] = values.squeeze()

                outputs = model(inputs, start)
                baseline = model(inputs, None)
                baseline = torch.gather(torch.softmax(baseline, dim=-1), -1, labels.unsqueeze(-1))
                normal = torch.gather(torch.softmax(outputs, dim=-1), -1, labels.unsqueeze(-1))
                rewards = normal - baseline
                probs, preds = torch.max(outputs, 1)
                if rl:
                    loss, policy_loss, entropy_loss = loss_func(action_probs, val,
                                                                            rewards, prob)
                else:
                    l = torch.arange(0, 196).repeat(bs, 1).to(device)
                    loss = criterion2(prob.permute(0, 2, 1), l)
                    entropy_loss = 0
                    policy_loss = 0


            else:

                bs, _, _, _ = inputs.shape
                start = torch.full((bs, 196), 196, dtype=torch.long, device=device)
                for i in range(196):
                    state = model.module.get_state(inputs, start)
                    actions, values = agent(state)

                    action = torch.argmax(actions, dim=-1)
                    start[:, i] = action
                outputs = model(inputs, start)
                probs, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        correct += torch.sum(preds == labels)
        n_items += inputs.size(0)
        if train_agent:
            running_loss += loss.item()
        else:
            running_loss += loss.item() * inputs.size(0)


        if counter % 1000 == 99:

            print(torch.argmax(prob[0], dim=-1))
            print(f'Reinforce_Loss {loss}')
            acc = correct / n_items
            print(acc)

        counter += 1
        del loss
        del outputs
    if train_agent:
        return running_loss, correct / n_items, policy_loss, entropy_loss
    return running_loss, correct / n_items



if __name__ == "__main__":
    set_deterministic()
    num_classes = 10
    max_epochs = 300
    base = None #"saves/best_rl_no_pretrain.pth"
    model = "Sequential"
    pretrained = False
    verbose = True
    agent = None#"saves/agent.pth"
    size = 224
    batch_size = 64
    use_simple_vit = False
    train(model, num_classes, max_epochs, base, reinforce=True, pretrained=pretrained,
          verbose=verbose, img_size=size, base_vit=use_simple_vit, batch_size = batch_size)