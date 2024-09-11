import torch
from .base_network import BaseNetwork
import torch.nn as nn

import sys
import os

# Add the path to the dinov2 directory
dinov2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dinov2'))
sys.path.append(dinov2_path)

from dinov2.models import build_model_from_cfg
from dinov2.configs import load_and_merge_config


class DINOV2Module(BaseNetwork):
    def __init__(self, args, mapping_dim=512):
        super().__init__(args)
        self.args = args
        self.device = args.device
        cfg = load_and_merge_config(self.args.dinov2_config_file)
        dino_encoder, encoder_embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        weights = torch.hub.load('facebookresearch/dinov2', self.args.dinov2_backbone)
        # load the weights
        dino_encoder.load_state_dict(weights.state_dict())
        self.dino_encoder = dino_encoder
        self.mapping = nn.Linear(encoder_embed_dim, mapping_dim)

        self.other_bias = nn.Parameter(torch.zeros(1))
        
    def simple_get_result_through_rest_layers(self, x: torch.Tensor, n: int = 1):
        for i, blk in enumerate(self.dino_encoder.blocks[-n:]):
            x = blk(x)
        x_norm = self.dino_encoder.norm(x)
        # directly return the cls token
        return x_norm[:, 0]

    def forward(self, x):
        x = self.simple_get_result_through_rest_layers(x, self.args.dinov2_train_transformer_block_to_last_index)
        x = self.mapping(x)
        return x
