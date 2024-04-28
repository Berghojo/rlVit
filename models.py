# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_H_14_Weights, VisionTransformer


class ViT(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        w = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        num_layers = 32
        patch_size = 8
        image_size = 32
        num_heads = 16
        hidden_dim = 1280
        mlp_dim = 5120
        super(ViT, self).__init__()
        self.backbone = VisionTransformer(image_size=image_size,
                                          patch_size=patch_size,
                                          num_layers=num_layers,
                                          num_heads=num_heads,
                                          hidden_dim=hidden_dim,
                                          mlp_dim=mlp_dim)
        if w:
            state_dict = w.get_state_dict(progress=False, check_hash=True)
            del state_dict["conv_proj.weight"]
            del state_dict["encoder.pos_embedding"]
            self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.heads.head = torch.nn.Linear(hidden_dim, num_classes)  # Used 768 from Documentation


    def forward(self, x):
        x = self.backbone(x).detach()
        return x

