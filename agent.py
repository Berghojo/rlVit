from torch import nn
from torchvision.models import resnet18
import torch
import random
from collections import namedtuple, deque

import numpy as np


class SimpleAgent(nn.Module):

    def __init__(self, n_patches):
        super(SimpleAgent, self).__init__()
        self.n_actions = n_patches+1
        self.action = nn.Linear(in_features=768, out_features=self.n_actions)
        self.value = nn.Linear(in_features=768, out_features=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6, norm=nn.LayerNorm(768))
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def freeze(self, freeze):
        print("Setting Agent Training to: ", freeze)
        for param in self.parameters():
            param.requires_grad = freeze

    def forward(self, state, memory=None, mask=None):
        if memory is None:
            memory = state
        t_mask = None #torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=state.device)

        x = self.decoder(state, memory, tgt_key_padding_mask=mask, tgt_is_causal=False, tgt_mask=t_mask)

        actor = self.action(x)
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
