import torch
from torch.cuda.amp import GradScaler
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
from torchvision.transforms import Resize, Normalize
import gc
from tqdm import tqdm
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

from util import CustomLoss
from agent import *
from models import *
from data import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'


    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html   Stand 2021
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def train(model_name, n_classes, max_epochs, base_model=None, reinforce=True, pretrained=True, agent_model=None,
          verbose=True, img_size=224, base_vit=False, batch_size=32, warmup=10, logging=10, use_baseline=False, alternate=True,
          rank=0, world_size=1):
    #torch.autograd.set_detect_anomaly(True)

    setup(rank, world_size)
    set_deterministic()
    if not os.path.exists("./saves"):
        os.makedirs("./saves/")
    device = rank #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_logging = logging
    pretraining_duration = warmup
    torch.cuda.empty_cache()
    gc.collect()
    launch_time = time.strftime("%Y_%m_%d-%H_%M")
    writer = SummaryWriter(log_dir='logs/' + model_name + launch_time)
    train_loader, test_loader = get_loader(img_size, batch_size)
    if reinforce:
        print("Reinforce")
        agent = SingleActionAgent(49)
        if agent_model is not None:
            new_state_dict = OrderedDict()
            mydic = torch.load(agent_model, map_location="cpu")
            ignore_list = []
            for k, v in mydic.items():
                name = k[7:] if k[:7] == "module." else k
                # remove `module.`x
                if name not in ignore_list:
                    new_state_dict[name] = v
            agent.load_state_dict(new_state_dict)

        agent = agent.to(device)
        agent = DDP(agent, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        agent_optimizer = optim.Adam(agent.parameters(), lr=1e-4)
        if (pretraining_duration > 0):
            agent_scheduler = optim.lr_scheduler.OneCycleLR(agent_optimizer, 1e-3, epochs=pretraining_duration,
                                                            steps_per_epoch=len(train_loader))

    else:
        agent = None
    if base_model:
        model = ViT(n_classes, device=device, pretrained=pretrained, reinforce=reinforce)
        mydic = torch.load(base_model, map_location="cpu")
        new_state_dict = OrderedDict()
        ignore_list = ["dark_patch"]
        for k, v in mydic.items():
            name = k[7:]  if k[:7] == "module." else k
                 # remove `module.`x
            if name not in ignore_list:
                new_state_dict[name] = v


        model.load_state_dict(new_state_dict)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    else:
        model = ViT(n_classes, device=device, pretrained=pretrained, reinforce=reinforce)

        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    class_accuracy, accuracy = eval_vit(model, device, test_loader, n_classes, agent if agent else None,
                                        verbose=verbose)
    print('[Test] ACC: {:.4f} '.format(accuracy))
    print(f'[Test] CLASS ACC: {class_accuracy} @-1')

    summarize(writer, "test", -1, accuracy)

    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, 1e-3, steps_per_epoch=len(train_loader),
                                              epochs=max_epochs-pretraining_duration)
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(max_epochs):
        if reinforce:
            if epoch == pretraining_duration:
                for g in agent_optimizer.param_groups:
                    g['lr'] = 1e-4
                agent_scheduler = optim.lr_scheduler.OneCycleLR(agent_optimizer, 1e-3,
                                                                epochs=max_epochs-pretraining_duration,
                                                                steps_per_epoch=len(train_loader))
                print("changed lr")
            if epoch < pretraining_duration:

                loss, acc = train_rl(train_loader, device, model,
                                     agent_optimizer, scaler, agent,
                                     train_agent=True, verbose=verbose, pretrain=True, use_baseline=use_baseline,
                                     scheduler=agent_scheduler)
                torch.save(agent.state_dict(), f"saves/base_{model_name}_agent_@{epoch}.pth")
                summarize(writer, "train", epoch, acc, loss)

            else:
                if alternate:
                    loss, acc, = train_rl(train_loader, device, model, model_optimizer, scaler, agent,
                                          train_agent=False,
                                          verbose=verbose, scheduler=scheduler)

                    summarize(writer, "train", epoch, acc, loss)
                agent_loss, agent_acc, policy_loss, value_loss, cum_reward = train_rl(train_loader, device, model,
                                                                                      agent_optimizer, scaler, agent,
                                                                                      train_agent=True, verbose=verbose,
                                                                                      pretrain=False,
                                                                                      use_baseline=use_baseline,
                                                                                      scheduler = agent_scheduler)


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
            print("saving agent")
            torch.save(agent.state_dict(), f"saves/best_{model_name}_agent_@{launch_time}.pth")


    cleanup()

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
    patches_per_side = 7
    with torch.no_grad():
        for inputs, labels in tqdm(loader, disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            if agent is not None:
                bs, _, _, _ = inputs.shape
                sequence = generate_max_agent(agent, bs, inputs, patches_per_side)
                outputs = model(inputs, sequence)
            else:
                outputs = model(inputs, None)
            probs, preds = torch.max(outputs, 1)

            for i, boolean in enumerate(preds == labels):
                overall[preds[i]] += 1
                if boolean:
                    correct[preds[i]] += 1
        if agent is not None:
            test_input, _ = next(iter(loader))
            test_input = torch.unsqueeze(test_input[0], 0)

            sequence = generate_max_agent(agent, 1, test_input, patches_per_side)
            f = open("permutation.txt", "a")

            print(list(sequence[0]), file=f)
            #print(list(probs), file=f)

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
        optimizer.zero_grad(set_to_none=True)
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


def train_rl(loader, device, model, optimizer, scaler, agent, train_agent, verbose=True, pretrain=False, use_baseline=False,
             scheduler=None):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if train_agent:
        #criterion2 = torch.nn.CrossEntropyLoss(ignore_index=196, label_smoothing=0.7)

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
    patches_per_side = 7
    if pretrain:
        batch_count = 0
        resize = Resize(35)
        value_criterion = torch.nn.MSELoss(reduction="mean")
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.4, reduction="none")
        for inputs, labels in tqdm(loader, disable=not verbose):

            optimizer.zero_grad(set_to_none=True)
            bs = inputs.shape[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_small = resize(inputs)
            sequence = torch.arange(0, 49, device=device, dtype=torch.long)
            sequence = sequence.repeat(bs, 1)
            random_idx = torch.randint(0, 48, (bs,), device=device)
            pseudo_labels = random_idx + 1

            state = torch.zeros_like(input_small, device=device)
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
            std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
            normalize = Normalize(mean=mean, std=std)
            state = normalize(state)
            unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())

            size = state.shape[2] // patches_per_side

            for i, idx in enumerate(random_idx):
                sequence[i, idx:] = 49
                if idx != 0:
                    r = (idx // patches_per_side) * size
                    c = (idx % patches_per_side) * size
                    state[i, :, :r, :] = input_small[i, :, :r, :].clone()
                    state[i, :, r:r+size, :c + size] = input_small[i, :, r:r+size, :c + size].clone()
            # for i in range(bs):
            #     plt.imshow(state[i].permute(1, 2, 0).cpu())
            #     plt.savefig(f"imgs/{i}.jpg")
            #     plt.imshow(input_small[i].permute(1, 2, 0).cpu())
            #     plt.savefig(f"imgs/og_{i}.jpg")



            values = torch.zeros((bs, 49), device=device)



            with (torch.amp.autocast(device_type="cuda", dtype=torch.float16)):
                logits, value = agent(state.detach())
                action_probs = torch.softmax(logits, dim=-1)
                probs, action = torch.max(action_probs, dim=-1)
                sequence[:, i] = action
                values[:, i] = value.squeeze()
            for img in range(bs):
                a = action[img]
                r = (a // patches_per_side) * size
                c = (a % patches_per_side) * size
                state[img, :, r:r+size, c:c+size] = input_small[img, :, r:r+size, c:c+size].clone()


            if batch_count % 100 == 99:
                image = unnormalize(state)
                plt.imshow(image[1].permute(1, 2, 0).cpu())
                plt.xlabel(action[1])
                plt.savefig(f"imgs/{i}.jpg")

            with torch.no_grad():
                outputs = model(inputs, sequence)
            probs, preds = torch.max(outputs, -1)
            rewards = (preds == labels).type(torch.half)

            value_loss = value_criterion(value.squeeze(), rewards)

            loss = criterion(logits, pseudo_labels).mean() + value_loss
            if batch_count % 100 == 99:
                print(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += (loss.item() * inputs.size(0))

            correct += torch.sum(preds == labels)
            n_items += inputs.size(0)
            batch_count += 1
        print("running_loss: ", running_loss)
        return running_loss, correct / n_items
    counter = 0
    if train_agent:
        loss_func = CustomLoss().to(device)
        cum_sum = 0
        p_loss = 0
        v_loss = 0
        for inputs, labels in tqdm(loader, disable=not verbose):
            optimizer.zero_grad(set_to_none=True)

            inputs = inputs.to(device)
            bs, _, _, _ = inputs.shape
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            preds, action_probs, model_probs, rewards, action_sequence, states = rl_training(agent, bs, inputs, labels, model, correct_only=not use_baseline)
            rewards = torch.stack(rewards, dim=0)
            cum_sum += torch.sum(rewards)

            old_states = torch.cat(states[:-1], dim=0)
            new_states = torch.cat(states[1:], dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

                actions, value = agent(old_states)


                actions = torch.softmax(actions, dim=-1)

                state_action_probs = actions.gather(1, action_sequence.flatten().unsqueeze(-1))

                gamma = 0.9
                pos_reward = 1
                neg_reward = 0
                rewards[rewards > 0] = pos_reward
                rewards[rewards <= 0] = neg_reward
                with torch.no_grad():
                    next_state_values = agent(new_states)[1].squeeze()

                k_step = 10
                gamma_tensor = torch.tensor([gamma ** k for k in range(k_step+1)], device=device)
                gamma_tensor = gamma_tensor.repeat(bs, 1)

                rewards = rewards.transpose(1, 0)
                discounted_rewards = torch.zeros_like(rewards, device=device)
                next_state_values = next_state_values.reshape(-1, bs).transpose(1, 0)

                for i in range(49):
                    if i < 49-k_step:
                        discounted_rewards[:, i] = torch.sum(torch.concat([rewards[:, i: i+k_step],
                                                                                     next_state_values[:, i+k_step].unsqueeze(1)], dim=-1) * gamma_tensor, dim=-1)
                    else:
                        discounted_rewards[:, i] = torch.sum(rewards[:, i:] * gamma_tensor[:, :49-i], dim=-1)

            loss, policy_loss, value_loss = loss_func(state_action_probs.squeeze(), value.squeeze(), discounted_rewards.flatten())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n_items += inputs.size(0)
            running_loss += loss.item()
            p_loss += policy_loss.item()
            v_loss += value_loss.item()
            correct += torch.sum(preds == labels)
            if counter % 100 == 99:
                print(torch.argmax(model_probs[0], dim=-1))
                print(f'Reinforce_Loss {loss}')
                acc = correct / n_items
                print(acc)
            counter += 1

        return running_loss, correct / n_items, p_loss, v_loss, cum_sum / counter
    else:
        for inputs, labels in tqdm(loader, disable=not verbose):
            inputs = inputs.to(device)
            bs, _, _, _ = inputs.shape
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            bs, _, _, _ = inputs.shape
            with torch.no_grad():
                sequence = generate_max_agent(agent, bs, inputs, patches_per_side)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs, sequence)
                outputs = torch.softmax(outputs, dim=-1)
                probs, preds = torch.max(outputs, -1)
            loss = criterion(outputs, labels)
            loss = torch.mean(loss)
            correct += torch.sum(preds == labels)

            n_items += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if counter % 100 == 99:
                print(f'Loss {loss}')
                acc = correct / n_items
                print(acc)
            counter += 1
            del loss
        return running_loss, correct / n_items


def generate_max_agent(agent, bs, inputs, patches_per_side):
    resize = Resize(35)
    input_small = resize(inputs)
    state = torch.zeros_like(input_small, device=inputs.device)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
    normalize = Normalize(mean=mean, std=std)
    state = normalize(state)
    sequence = torch.full((bs, 49), 49, device=inputs.device, dtype=torch.long)
    values = torch.zeros((bs, 49), device=inputs.device)
    size = state.shape[2] // patches_per_side
    for i in range(49):
        logits, value = agent(state.detach())
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        sequence[:, i] = action

        values[:, i] = value.squeeze()
        for img in range(bs):
            a = action[img]
            r = (a // patches_per_side) * size
            c = (a % patches_per_side) * size
            state[img, :, c:c + size, r:r + size] = input_small[img, :, c:c + size, r:r + size].clone()
    return sequence


def rl_training(agent, bs, inputs, labels, model, correct_only=False):
    with torch.no_grad():
        resize = Resize(35)
        patches_per_side = 7
        bs = inputs.shape[0]

        input_small = resize(inputs)
        state = torch.zeros_like(input_small, device=inputs.device)

        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
        normalize = Normalize(mean=mean, std=std)
        state = normalize(state)

        states = []
        rewards = []
        og_baseline = model(inputs, None).detach()
        baseline = torch.gather(torch.softmax(og_baseline, dim=-1), -1, labels.unsqueeze(-1))
        states.append(state)
        sequence = torch.full((bs, 49), 49,  device=inputs.device, dtype=torch.long)
        sequence_probs = torch.zeros((bs, 49, 50), device=inputs.device)
        values = torch.zeros((bs, 49), device=inputs.device)
        size = state.shape[2] // patches_per_side
        for i in range(49):
            outputs = model(inputs, sequence)
            probs, preds = torch.max(outputs, -1)

            logits, value = agent(state.detach())
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            sequence_probs[:, i] = action_probs
            sequence[:, i] = action

            values[:, i] = value.squeeze()
            for img in range(bs):
                a = action[img]
                r = (a // patches_per_side) * size
                c = (a % patches_per_side) * size
                state[img, :, c:c + size, r:r + size] = input_small[img, :, c:c + size, r:r + size].clone()
            states.append(state)




            if correct_only:
                reward = (preds == labels).long()
                rewards.append(reward.squeeze())
            else:
                normal = torch.gather(torch.softmax(outputs, dim=-1), -1, labels.unsqueeze(-1))
                reward_mask = (preds == labels).long()

                reward = (normal - baseline).squeeze() * reward_mask
                #reward[reward == 0] = 0.001 #Equality to baseline should be rewarded?
                rewards.append(reward)

        return preds, sequence_probs, probs, rewards, sequence, states


def combine_to_batch(bs, inputs, labels, model, old_action):
    inputs = inputs.repeat(49, 1, 1, 1)
    subactions = []
    for i in range(49):
        #mask = torch.cat([torch.zeros((bs, i + 1)), torch.ones((bs, 49 - i - 1))], dim=-1).bool()
        sub_action = old_action.clone()
        #sub_action[mask] = 49
        subactions.append(sub_action)
    subactions = torch.cat(subactions, dim=0)
    outputs = model(inputs, subactions)
    probs, preds = torch.max(outputs, 1)
    del outputs
    reward = (preds == labels.repeat(49)).long()
    rewards = reward.reshape((bs, 49))
    preds = preds.reshape((bs, 49))[:, -1]
    probs = probs.reshape((bs, 49))[:, -1]
    return preds, probs, rewards


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
    batch_size = 2
    use_simple_vit = False
    train(model, num_classes, max_epochs, base, reinforce=True, pretrained=pretrained,
          verbose=verbose, img_size=size, base_vit=use_simple_vit, batch_size=batch_size)
