from torch import nn
from torchvision.models import resnet18
import torch
from util import PositionalEncoding1D

from copy import deepcopy
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

