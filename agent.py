from torch import nn
from torchvision.models.vision_transformer import vit_b_16
import torch

from copy import deepcopy
class Agent(nn.Module):
    def __init__(self, n_patches, pretrained):
        super(Agent, self).__init__()
        self.n_actions = n_patches+1
        self.hidden_dim = 768
        back = vit_b_16(pretrained=True)
        self.backbone = deepcopy(back.pretrained)
        self.head1 = nn.Linear(in_features=768, out_features=self.n_actions)
        self.head2 = (nn.Linear(in_features=768, out_features=1))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.proj_layer = back.conv_proj
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

    def forward(self, x):
        x = self._process_input(x, 16)
        x = self.backbone(x)
        table = self.softmax(self.head1(x[:, 1:]))
        values = self.head2(x)
        return table, values
