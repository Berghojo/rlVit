from torch import nn
from torchvision.models.vision_transformer import vit_b_16
import torch

from copy import deepcopy
class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 768
        back = vit_b_16(pretrained=pretrained)
        #self.backbone = deepcopy(back.encoder)
        self.linear1 = nn.Linear(self.hidden_dim, 384)
        self.head1 = nn.Linear(in_features=384, out_features=self.n_actions)
        self.head2 = (nn.Linear(in_features=384, out_features=1))
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.proj_layer = deepcopy(back.conv_proj)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=False)

    def _process_input(self, x: torch.Tensor, p: int) -> torch.Tensor:
        n, c, h, w = x.shape

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.proj_layer(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1).detach()
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def freeze(self, freeze):
        for param in self.parameters():
            param.requires_grad = freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.proj_layer.parameters():
            param.requires_grad = False

    def copy_backbone(self, state_dict):
        self.proj_layer.weight.data = state_dict["module.proj_layer.weight"]
        self.proj_layer.bias.data = state_dict["module.proj_layer.bias"]


    def forward(self, x, ps=16):
        x = self._process_input(x, ps)
        x = self.relu(self.linear1(x))
        table = self.softmax(self.head1(x[:, 1:]))
        values = self.head2(x[:, 1:])
        return table, values
