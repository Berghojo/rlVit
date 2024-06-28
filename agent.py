from torch import nn
from torchvision.models import resnet18
import torch
from util import PositionalEncoding1D

from copy import deepcopy
class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 1024

        self.linear1 = nn.Linear(512, self.hidden_dim)
        self.head1 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.head2 = (nn.Linear(in_features=self.hidden_dim, out_features=1))
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.query_embed = nn.Embedding(n_patches, self.hidden_dim)
        resnet = resnet18(pretrained=True)  # pretrained ImageNet resnet34
        self.dropout = nn.Dropout(p=0.5)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.decode_length = 196
        decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        for param in self.resnet.parameters():
            param.requires_grad = False
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
        x = torch.flatten(x, 2)
        x = torch.permute(x, [0, 2, 1])
        x = self.relu(self.linear1(x)).permute(1, 0, 2)

        _, bs, f = x.shape
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_pos).permute(1, 0, 2)

        x = self.transformer_decoder(tgt, x).permute(1, 0, 2)

        tbl = self.relu(self.head1(x))
        value = self.head2(x)
        return self.softmax(tbl), value

