from torch import nn
from torchvision.models.vision_transformer import vit_b_16
import torch

from copy import deepcopy
class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 768
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding="same")
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same")
        self.linear1 = nn.Linear(768, 128)
        self.head1 = nn.Linear(in_features=128, out_features=self.n_actions)
        self.head2 = (nn.Linear(in_features=128, out_features=1))
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.proj_layer = nn.Conv2d(
            in_channels=3, out_channels=768, kernel_size=16, stride=16
        )
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

        return x

    def freeze(self, freeze):
        print("Setting Agent Training to: ", freeze)
        for param in self.parameters():
            param.requires_grad = freeze


    def copy_backbone(self, state_dict):
        pass


    def forward(self, x):
        x = self._process_input(x, 16)
        x = self.relu(self.linear1(x))
        table = self.softmax(self.head1(x))
        values = self.head2(x)
        return table, values