# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights, VisionTransformer, EncoderBlock
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
from torch import nn
from collections import OrderedDict
from copy import deepcopy

class baseViT(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False, device=None):
        super(baseViT, self).__init__()
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        image_size = 224
        patch_size = 16
        num_layers = 12
        num_heads = 12
        hidden_dim = 768
        mlp_dim = 3072
        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim
        )

        if w:
            state_dict = w.get_state_dict(progress=False, check_hash=True)
            del state_dict["encoder.pos_embedding"]
            self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.heads.head = torch.nn.Linear(hidden_dim, num_classes)  # Used 768 from Documentation

    def forward(self, input):
        x = self.backbone(input)
        return x


class ViT(torch.nn.Module):
    def __init__(self, num_classes, device=None):
        super(ViT, self).__init__()
        image_size = 224
        self.patch_sizes = [16, 32]
        num_layers = [12, 8]
        seq_lens = [(image_size // patch_size) ** 2 + 1 for patch_size in self.patch_sizes]

        num_heads = 12
        dropout = 0.0
        attention_dropout = 0.0
        self.hidden_dims = [768, int(768 * 2)]
        mlp_dim = 3072
        proj_layers = [nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=size, stride=size
        ) for hidden_dim, size in zip(self.hidden_dims, self.patch_sizes)]
        self.proj_layers = nn.ModuleList(proj_layers)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embedding = [nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)) for
                              seq_length, hidden_dim in zip(seq_lens, self.hidden_dims)]  # from BERT
        self.pos_embedding = nn.ParameterList(self.pos_embedding)

        self.class_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, hd)) for hd in self.hidden_dims])

        self.downsample = nn.Sequential(nn.Conv2d(self.hidden_dims[0], self.hidden_dims[0], kernel_size=2,
                                                  stride=2,
                                                  dilation=1,
                                                  padding=0,
                                                  groups=self.hidden_dims[0],
                                                  bias=False,
                                                  ),
                                        nn.BatchNorm2d(self.hidden_dims[0]),
                                        nn.Conv2d(
                                            self.hidden_dims[0],
                                            self.hidden_dims[1],
                                            kernel_size=1,
                                            stride=1,
                                            dilation=1,
                                            padding=0,
                                            groups=1,
                                            bias=True,
                                        ),
                                        nn.BatchNorm2d(self.hidden_dims[1]),
                                        )
        self.upsample = nn.Sequential(nn.Conv2d(
            self.hidden_dims[1],
            self.hidden_dims[0],
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            groups=1,
            bias=True,
        ),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            )
        )
        self.upsample2 = deepcopy(self.upsample)
        self.downsample2 = deepcopy(self.downsample)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layers: OrderedDict[str, nn.Module] = OrderedDict()
        for e, hidden_dim in enumerate(self.hidden_dims):
            for i in range(num_layers[e]):
                self.layers[f"encoder_layer_{e}_{i}"] = EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                ).to(device)

        self.head = torch.nn.Linear(sum(self.hidden_dims), num_classes)  # Used 768 from Documentation

    def _process_input(self, x: torch.Tensor, p: int, i) -> torch.Tensor:
        n, c, h, w = x.shape

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.proj_layers[i](x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dims[i], n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, input):
        x = self._process_input(input, self.patch_sizes[0], 0)
        x2 = self._process_input(input, self.patch_sizes[1], 1)

        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token[0].expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding[0]

        for i in range(4):
            x = self.layers[f"encoder_layer_0_{i}"](x)

        x1 = x.permute(0, 2, 1)
        patch_size = x1[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)
        x1 = x1[:, :, 1:].reshape(n, -1, patch_size, patch_size)

        x1 = self.downsample(x1)
        x1 = x1.reshape(n, self.hidden_dims[1], -1)
        x1 = x1.permute(0, 2, 1)
        batch_class_token = self.class_token[1].expand(n, -1, -1)

        x1 = torch.cat([batch_class_token, x1], dim=1)
        x1 = x1 + self.pos_embedding[1]

        for i in range(4):
            x1 = self.layers[f"encoder_layer_1_{i}"](x1)
            x = self.layers[f"encoder_layer_0_{i+4}"](x)

        x, x1 = self.fuse(x, x1)

        for i in range(4, 8):
            x1 = self.layers[f"encoder_layer_1_{i}"](x1)
            x = self.layers[f"encoder_layer_0_{i+4}"](x)

        x, x1 = self.fuse(x, x1)
        x = x[:, 0]
        return x

    def fuse(self, x1, x2):
        n = x1.shape[0]
        down = x1.permute(0, 2, 1)
        patch_size = down[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)

        down = down[:, :, 1:].reshape(n, -1, patch_size, patch_size)

        down = self.downsample2(down)

        down = down.reshape(n, self.hidden_dims[1], -1)
        down = down.permute(0, 2, 1)



        up = x2.permute(0, 2, 1)
        patch_size = up[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)

        up = up[:, :, 1:].reshape(n, -1, patch_size, patch_size)
        up = self.upsample2(up)

        up = up.reshape(n, self.hidden_dims[0], -1)
        up = up.permute(0, 2, 1)


        x2[:, 1:] = self.act(x2[:, 1:]+down)
        x1[:, 1:] = self.act(x1[:, 1:]+up)


        return x1, x2












class Agent(torch.nn.Module):
    def __init__(self, n_patches):
        super(Agent, self).__init__()
        actor = n_patches ** 2
