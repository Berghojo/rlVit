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


def train(model_name, n_classes, max_epochs, base_model=None, reinforce=True, pretrained=False, agent_model=None,
          verbose=True, img_size=224, base_vit=False, batch_size=32, warmup=10, logging=10, use_baseline=False):
    #torch.autograd.set_detect_anomaly(True)

    if not os.path.exists("./saves"):
        os.makedirs("./saves/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_logging = logging
    pretraining_duration = warmup
    torch.cuda.empty_cache()
    gc.collect()
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)
    train_loader, test_loader = get_loader(img_size, batch_size)
    if reinforce:
        print("Reinforce")
        agent = SimpleAgent(49)
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
        # class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes, None, verbose=verbose)
        # print('[Test] ACC: {:.4f} '.format(accuracy))
        # print(f'[Test] CLASS ACC: {class_accuracy} @-1')
        #
        # summarize(writer, "test", -1, accuracy)
    else:
        model = ViT(n_classes, device=device, pretrained=pretrained, reinforce=reinforce) if not base_vit else BaseVit(
            10, pretrained)
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, max_epochs)
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(max_epochs):
        if reinforce:
            if epoch == pretraining_duration:
                for g in agent_optimizer.param_groups:
                    g['lr'] = 1e-8

            if epoch < pretraining_duration:

                train_rl(train_loader, device, model,
                         agent_optimizer, scaler, agent,
                         train_agent=True, verbose=verbose, pretrain=True, use_baseline=use_baseline)

            else:
                # loss, acc, = train_rl(train_loader, device, model, model_optimizer, scaler, agent, train_agent=False,
                #                       verbose=verbose)
                #
                # summarize(writer, "train", epoch, acc, loss)
                agent_loss, agent_acc, policy_loss, value_loss, cum_reward = train_rl(train_loader, device, model,
                                                                                      agent_optimizer, scaler, agent,
                                                                                      train_agent=True, verbose=verbose,
                                                                                      pretrain=False, use_baseline=use_baseline)


                summarize_agent(writer, "train_agent", epoch, cum_reward,  value_loss, policy_loss)
                summarize(writer, "train_agent", epoch, agent_acc, agent_loss)


            #summarize(writer, "train", epoch, acc, loss)
        else:
            loss, acc = train_vit(train_loader, device, model, model_optimizer, scaler, verbose=verbose)
            summarize(writer, "train", epoch, acc, loss)
        if epoch >= start_logging:
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


def summarize_agent(writer, split, epoch, cum_reward, value_loss, policy_loss):
    writer.add_scalar('cum_reward/' + split, cum_reward, epoch)
    writer.add_scalar('value_loss/' + split, value_loss, epoch)
    writer.add_scalar('policy_loss/' + split, policy_loss, epoch)


def eval_vit(model, device, loader, n_classes, agent, verbose=True):
    model.eval()
    if agent is not None:
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
                start = torch.full((bs, 49), 49, dtype=torch.long, device=device)
                start[:, 0] = 0
                initial = torch.arange(0, 49, device=device)
                initial = initial.repeat(bs, 1)
                image = model.module.get_state(inputs.to(device), initial)
                for i in range(48):
                    state = model.module.get_state(inputs, start)
                    mask = start == 49
                    actions, values = agent(state, image, mask)
                    actions = torch.softmax(actions, dim=-1)
                    vals, action = torch.max(actions, dim=-1)

                    start[:, i+1] = action[:, i]

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
        start = torch.full((1, 49), 49, dtype=torch.long, device=device)
        probs = torch.full((1, 49), 0, dtype=torch.long, device=device)
        initial = torch.arange(0, 49, device=labels.device)
        initial = initial.repeat(1, 1)
        image = model.module.get_state(test_input.to(device), initial.to(device))

        start[:, 0] = 0
        for i in range(48):
            state = model.module.get_state(test_input.to(device), start)

            mask = start == 49
            actions, values = agent(state, image, mask)

            vals, action = torch.max(actions, dim=-1)
            start[:, i+1] = action[:, i]
            probs[:, i+1] = vals[:, i]
        f = open("permutation.txt", "a")

        print(list(start), file=f)

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
        if counter % 100 == 99:
            print(correct / n_items)
        counter += 1

    return running_loss, correct / n_items


def train_rl(loader, device, model, optimizer, scaler, agent, train_agent, verbose=True, pretrain=False, use_baseline=False):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if train_agent:
        exp_replay = ReplayMemory(10000)

        #criterion2 = torch.nn.CrossEntropyLoss(ignore_index=196, label_smoothing=0.7)
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
    if pretrain:

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.7, reduction="none")
        batch_count = 0
        for inputs, labels in tqdm(loader, disable=not verbose):
            batch_count += 1
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            bs, _, _, _ = inputs.shape
            optimizer.zero_grad()
            start = torch.arange(0, 49, device=device)
            start = start.repeat(bs, 1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                state = model.module.get_state(inputs, start).detach()
                actions, _ = agent(state)
            pseudo_labels = torch.arange(1, 50, device=device)
            pseudo_labels = pseudo_labels.repeat(bs, 1)

            loss = criterion(actions.flatten(0, 1), pseudo_labels.flatten(0, 1))
            loss = torch.mean(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            with torch.no_grad():
                outputs = model(inputs, start)
            probs, preds = torch.max(outputs, -1)
            correct += torch.sum(preds == labels)
            n_items += inputs.size(0)
        print("running_loss: ", running_loss)
        return running_loss, correct / n_items
    if train_agent:
        batch_count = 0
        cum_sum = 0
        p_loss = 0
        v_loss = 0
        for inputs, labels in tqdm(loader, disable=not verbose):
            batch_count += 1
            inputs = inputs.to(device)
            bs, _, _, _ = inputs.shape
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad()

            preds, prob, probs, rewards = rl_training(agent, bs, exp_replay, inputs, labels, model, correct_only = not use_baseline)
            cum_sum += torch.sum(rewards)
            batchsize = 32
            if len(exp_replay) > 100:
                while len(exp_replay) > batchsize:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        state_batch, action_batch, reward_batch, next_state_batch = exp_replay.sample(batchsize)
                        state_batch = state_batch.to(device)

                        action_batch = action_batch.to(device)
                        reward_batch = reward_batch.to(device)


                        next_state_batch = next_state_batch.to(device)

                        action, value = agent(state_batch)
                        action = torch.softmax(action, dim=-1)
                        state_action_values = torch.softmax(action, dim=-1).gather(2, action_batch.unsqueeze(-1))

                        gamma = 0.9
                        pos_reward = 1
                        neg_reward = -0.01


                        reward_batch[reward_batch > 0] = pos_reward
                        reward_batch[reward_batch <= 0] = neg_reward

                        next_state_values = torch.zeros_like(value, device=device).half().squeeze()

                        with torch.no_grad():
                            next_state_values[:, :-1] = agent(next_state_batch[:, :-1])[1].squeeze()
                        expected_state_action_values = (next_state_values * gamma) + reward_batch
                        loss, policy_loss, value_loss = loss_func(state_action_values.squeeze(), value.squeeze(), expected_state_action_values)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    n_items += inputs.size(0)
                    running_loss += loss.item()
                    p_loss += policy_loss.item()
                    v_loss += value_loss.item()
                    correct += torch.sum(preds == labels)
                    if counter % 100 == 99:
                        print(torch.argmax(probs[0], dim=-1))
                        print(f'Reinforce_Loss {loss}')
                        acc = correct / n_items
                        print(acc)
                    counter += 1

        return running_loss, correct / n_items, p_loss, v_loss, cum_sum / batch_count
    else:
        for inputs, labels in tqdm(loader, disable=not verbose):
            inputs = inputs.to(device)
            bs, _, _, _ = inputs.shape
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad()
            bs, _, _, _ = inputs.shape
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                start = torch.full((bs, 49), 49, dtype=torch.long, device=device)
                initial = torch.arange(0, 49, device=labels.device)
                initial = initial.repeat(bs, 1)
                image = model.module.get_state(inputs.to(device), initial)
                start[:, 0] = 0
                for i in range(48):
                    mask = start == 49

                    state = model.module.get_state(inputs, start)

                    actions, values = agent(state, image, mask)
                    actions = torch.softmax(actions, dim=-1)
                    action = torch.argmax(actions, dim=-1)
                    start[:, i+1] = action[:, i]
                outputs = model(inputs, start)
                probs, preds = torch.max(outputs, -1)

            loss = criterion(outputs, labels)
            loss = torch.mean(loss)
            correct += torch.sum(preds == labels)
            n_items += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if counter % 100 == 99:
                print(torch.max(probs[0], dim=-1))
                print(f'Loss {loss}')
                acc = correct / n_items
                print(acc)
            counter += 1
            del loss
        return running_loss, correct / n_items


def rl_training(agent, bs, exp_replay, inputs, labels, model, correct_only=False):
    with torch.no_grad():
        start = torch.arange(0, 49, device=labels.device)
        #start = torch.full((bs, 49), 49, dtype=torch.long, device=labels.device)
        start = start.repeat(bs, 1)
        state = model.module.get_state(inputs, start).detach()
        actions, _ = agent(state)
        prob = torch.softmax(actions, dim=-1)
        dist = Categorical(prob)
        action = dist.sample()
        state = model.module.get_state(inputs, action).detach()
        action = action[:, 1:]
        state[:, -1, :] = float('nan')
        action = torch.cat([torch.zeros((bs, 1), device=labels.device), action], dim=1).long()

        old_state = model.module.get_state(inputs, action).detach()

        #exp_replay.push(list(old_state.to("cpu")), list(action.flatten(1).to("cpu")), list(state.to("cpu")), [0] * bs)
        rewards = torch.zeros((bs,49), device=labels.device)
        outputs = model(inputs, action).detach()
        probs, preds = torch.max(outputs, 1)

        if correct_only:
            reward = (preds == labels).long()
            rewards[:, -1] = reward
            exp_replay.push(list(old_state.to("cpu")), list(action.to("cpu")),
                            list(state.to("cpu")), list(rewards.to("cpu")))

        else:
            og_baseline = model(inputs, None).detach()
            baseline = torch.gather(torch.softmax(og_baseline, dim=-1), -1, labels.unsqueeze(-1))
            for i in range(49):

                mask = torch.cat([torch.zeros((bs, i+1)), torch.ones((bs, 49-i-1))], dim=-1).bool()
                sub_action = action.clone()
                sub_action[mask] = 49
                outputs = model(inputs, action).detach()
                normal = torch.gather(torch.softmax(outputs, dim=-1), -1, labels.unsqueeze(-1))


                reward = (normal - baseline)
                reward[reward == 0] = 0.001 #Equality to baseline should be rewarded
                rewards[:, i] = reward.squeeze()
            exp_replay.push(list(old_state.to("cpu")), list(action.to("cpu")),
                            list(state.to("cpu")), list(rewards.to("cpu")))
        return preds, prob, probs, rewards

if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    set_deterministic()
    num_classes = 10
    max_epochs = 300
    base = "saves/bestr.pth"
    model = "New Approach"
    pretrained = False
    verbose = True
    agent = None  #"saves/agent.pth"

    size = 224
    batch_size = 32
    use_simple_vit = False
    train(model, num_classes, max_epochs, base, reinforce=True, pretrained=pretrained,
          verbose=verbose, img_size=size, base_vit=use_simple_vit, batch_size=batch_size)
