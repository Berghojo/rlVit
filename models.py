# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights, VisionTransformer, EncoderBlock
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
from torch import nn
import numpy as np
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


class ParallelEncoder(nn.Module):
    def __init__(self, stage, num_heads, hidden_dims, mlp_dim, dropout, attention_dropout, norm_layer):
        super(ParallelEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        for e, layers in enumerate(stage):
            encoder = OrderedDict()
            for l in range(layers):
                encoder[f'encoder_{e}_{l}'] = EncoderBlock(
                    num_heads,
                    hidden_dims[e],
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
            self.encoder_blocks.append(nn.Sequential(encoder))

    def forward(self, x):
        for stream, encoder in enumerate(self.encoder_blocks):
            x[stream] = encoder(x[stream])

        return x


class FusionLayer(nn.Module):
    def __init__(self, stage, stream_dims, pos_embedding, class_token):
        super(FusionLayer, self).__init__()
        self.stage = stage
        self.transitions = nn.ModuleList()
        self.pos_embedding = pos_embedding
        self.class_token = class_token
        self.act = nn.GELU()
        self.stream_dims = stream_dims
        for i in range(len(stage)):
            first = nn.ModuleList()
            for j in range(len(stage)):
                if i == j:
                    first.append(nn.Identity())
                elif i < j:
                    first.append(nn.Sequential(nn.Conv2d(stream_dims[i], stream_dims[i], kernel_size=2 * (j - i),
                                                         stride=2 * (j - i),
                                                         dilation=1,
                                                         padding=0,
                                                         groups=stream_dims[i],
                                                         bias=False,
                                                         ),
                                               nn.BatchNorm2d(stream_dims[i]),
                                               nn.Conv2d(
                                                   stream_dims[i],
                                                   stream_dims[j],
                                                   kernel_size=1,
                                                   stride=1,
                                                   dilation=1,
                                                   padding=0,
                                                   groups=1,
                                                   bias=True,
                                               ),
                                               nn.BatchNorm2d(stream_dims[j]),
                                               ))
                else:
                    first.append(nn.Sequential(nn.Conv2d(
                        stream_dims[i],
                        stream_dims[j],
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        padding=0,
                        groups=1,
                        bias=True,
                    ),
                        nn.BatchNorm2d(stream_dims[j]),
                        nn.Upsample(
                            scale_factor=2 * (i - j),
                            mode="nearest"
                        )
                    ))
            self.transitions.append(first)

    def forward(self, x):
        n_streams = len(self.stage)
        streams = []
        n = x[0].shape[0]
        for stream in range(len(x)):
            img = self.reimage(x[stream])

            outputs = []
            for t, transition in enumerate(self.transitions[stream]):

                trans = transition(img)
                outputs.append(trans)

            streams.append(outputs)

        summed = streams[0]
        for s in streams:
            for i in range(1, n_streams):
                summed[i] += s[i]

        for i in range(n_streams):
            #
            summed[i] = summed[i].reshape(n, self.stream_dims[i], -1)
            summed[i] = summed[i].permute(0, 2, 1)
            activated = nn.functional.gelu(summed[i].clone())
            if i >= len(x):
                batch_class_token = self.class_token[i].expand(n, -1, -1)
                activated = torch.cat([batch_class_token, activated], dim=1)

                new = activated + self.pos_embedding[i]
                x.append(new)
            else:

                x[i][:, 1:] = activated
        return x


    def reimage(self, x):
        n = x.shape[0]
        x = x.permute(0, 2, 1)
        patch_size = x[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)
        x = x[:, :, 1:].reshape(n, -1, patch_size, patch_size)
        return x


class ViT(torch.nn.Module):
    def __init__(self, num_classes, device=None, img_size=248):
        super(ViT, self).__init__()
        image_size = img_size
        self.patch_sizes = [8, 16, 32]
        self.stages = ((4,),
                       (4, 4),
                       (4, 4, 4))

        seq_lens = [(image_size // patch_size) ** 2 + 1 for patch_size in self.patch_sizes]
        base_size = 384
        num_heads = 12
        dropout = 0.1
        attention_dropout = 0.1
        self.hidden_dims = [base_size] + [int(base_size * 2 * i) for i in range(1, len(self.patch_sizes)
                                                                                )]
        print(self.hidden_dims)
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

        self.dropout = nn.Dropout(dropout)
        self.parallel_encoders = nn.ModuleList(
            [ParallelEncoder(s, num_heads, self.hidden_dims, mlp_dim, dropout, attention_dropout, norm_layer) for s in
             self.stages])
        self.fusion_layers = nn.ModuleList([FusionLayer(s, self.hidden_dims, self.pos_embedding, self.class_token) for s in self.stages[1:]])

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

        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token[0].expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding[0]
        x = [x]
        for e, stage in enumerate(self.stages):
            x = self.parallel_encoders[e](x)
            if e != len(self.stages) - 1:
                x = self.fusion_layers[e](x)

        x = torch.cat([el[:, 0] for el in x], dim=1)
        x = self.head(x)
        return x

    def fuse(self, x1, x2, iteration):
        n = x1.shape[0]
        down = x1.permute(0, 2, 1)
        patch_size = down[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)

        down = down[:, :, 1:].reshape(n, -1, patch_size, patch_size)

        down = self.downsamples[iteration](down)

        down = down.reshape(n, self.hidden_dims[1], -1)
        down = down.permute(0, 2, 1)

        up = x2.permute(0, 2, 1)
        patch_size = up[:, :, 1:].shape[2]
        patch_size = int((patch_size) ** 0.5)

        up = up[:, :, 1:].reshape(n, -1, patch_size, patch_size)
        up = self.upsamples[iteration](up)

        up = up.reshape(n, self.hidden_dims[0], -1)
        up = up.permute(0, 2, 1)

        x2[:, 1:] = self.act(x2[:, 1:] + down)
        x1[:, 1:] = self.act(x1[:, 1:] + up)

        return x1, x2


class Agent(torch.nn.Module):
    def __init__(self, n_patches):
        super(Agent, self).__init__()
        actor = n_patches ** 2
