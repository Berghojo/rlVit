from torch import nn
from torchvision.models import resnet18
import torch
import random
from torch import Tensor, device, dtype
from collections import namedtuple, deque

import numpy as np


class SingleActionAgent(nn.Module):
    def __init__(self, n_patches):
        super(SingleActionAgent, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding="same")
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same")
        self.fc = nn.Linear(35 * 35 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.lstm = nn.LSTMCell(256, 128)
        self.action = nn.Linear(128, n_patches+1)
        self.value = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self._reset_parameters()
        self.dout = nn.Dropout(p=0.5)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def freeze(self, freeze):
        print("Setting Agent Training to: ", freeze)
        for param in self.parameters():
            param.requires_grad = freeze

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training),
            torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training)
        )
    def forward(self, x: Tensor, hidden) -> Tensor:
        x = self.dout(self.conv(x))
        x = self.dout(self.conv_2(x))
        x = self.relu(x.flatten(1))

        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        h_0, c_0 = hidden
        x, c_x = hidden = self.lstm(x, (h_0.detach(), c_0.detach()))
        action = self.action(x)
        value = self.value(x)
        return action, value, hidden

