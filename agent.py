from torch import nn
from torchvision.models import resnet18
import torch
import random
from collections import namedtuple, deque

import numpy as np

class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 256
        self.head1 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.head2 = (nn.Linear(in_features=self.hidden_dim, out_features=1))
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.query_embed = nn.Embedding(n_patches, self.hidden_dim)
        resnet = resnet18(pretrained=True)  # pretrained ImageNet resnet34
        self.dropout = nn.Dropout(p=0.5)

        self.decode_length = 196

        decoder_norm = nn.LayerNorm(self.hidden_dim)
        encoder_norm = nn.LayerNorm(self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, norm_first=True, batch_first=False)
        self.input_proj = nn.Conv2d(512, self.hidden_dim, kernel_size=1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6, norm=decoder_norm)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, norm_first=True, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)

        self._reset_parameters()

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x

    def freeze(self, freeze):
        print("Setting Agent Training to: ", freeze)
        for param in self.parameters():
            param.requires_grad = freeze


    def copy_backbone(self, state_dict):
        pass


    def forward(self, x):
        x = self.dropout(self.relu(self.resnet(x)))
        x = self.input_proj(x)
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        memory = self.transformer_encoder(x)
        x = self.transformer_decoder(tgt, memory).permute(1, 0, 2)


        tbl = self.head1(x)

        value = self.head2(x)
        return self.softmax(tbl), value

class SimpleAgent(nn.Module):

    def __init__(self, n_patches):
        super(SimpleAgent, self).__init__()
        self.n_actions = n_patches+1
        self.linear1 = nn.Linear(in_features=768, out_features=128)
        self.action= nn.Linear(in_features=128, out_features=self.n_actions)
        self.value = nn.Linear(in_features=128, out_features=1)
        self.softmax = nn.LogSoftmax(dim=-1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        self.relu = nn.ReLU()
    def freeze(self, freeze):
        print("Setting Agent Training to: ", freeze)
        for param in self.parameters():
            param.requires_grad = freeze

    def forward(self, state, mask=None):
        x = self.decoder(state, state, tgt_key_padding_mask=mask)
        x = self.relu(self.linear1(x))
        actor = self.softmax(self.action(x))
        critic = self.value(x)
        return actor, critic
        # state = state.flatten(1)
        # x = self.relu(self.linear1(state))
        # actor = self.softmax(self.action(x))
        # critic = self.value(x)
        # return actor, critic


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.states = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.next_states = deque([], maxlen=capacity)

    def push(self, states, actions, next_states, rewards):
        """Save a transition"""
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_states.extend(next_states)

    def sample(self, batch_size):
        sample = random.sample(range(len(self)), batch_size)

        s = torch.stack(list(self.states))[sample]
        a = torch.stack(list(self.actions))[sample]
        r = torch.stack(list(self.rewards))[sample]
        ns = torch.stack(list(self.next_states))[sample]
        sample.sort(reverse=True)

        for i in sample:
            del self.states[i]
            del self.actions[i]
            del self.rewards[i]
            del self.next_states[i]

        return s, a, r, ns

    def __len__(self):
        assert len(self.states) == len(self.actions) and len(self.rewards) == len(self.states) and len(self.next_states) == len(self.states)
        return len(self.states)