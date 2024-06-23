from torch import nn
from torchvision.models import resnet34
import torch

from copy import deepcopy
class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 1024

        self.linear1 = nn.Linear(25088, self.hidden_dim)
        self.head1 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.head2 = (nn.Linear(in_features=self.hidden_dim, out_features=1))
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

        resnet = resnet34(pretrained=True)  # pretrained ImageNet resnet34

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.decode_length = 196
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=False)

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
        b, n, h, w = x.shape
        x = self.resnet(x).detach()
        x = torch.flatten(x, 1)

        x = self.relu(self.linear1(x))
        h = torch.zeros((b, self.hidden_dim)).to(x.device)
        c = torch.zeros((b, self.hidden_dim)).to(x.device)
        output = torch.zeros((b, self.decode_length, self.hidden_dim)).to(x.device)
        for i in range(self.decode_length):
            h, c = self.lstm(x, (h, c))
            output[:, i] = h
        table = self.head1(output)
        values = self.head2(output)
        return table, values