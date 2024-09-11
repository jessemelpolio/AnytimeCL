import torch
from .base_network import BaseNetwork
from .clip_module import LearnableCLIPModule
from .memory_module import MemoryModule


class FullModel(BaseNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        self.clip_branch = LearnableCLIPModule(args)
        self.clip_encoder = self.clip_branch.clip_encoder

        self.retrieval_branch = MemoryModule(args, self.clip_branch)
        self.runtime_get_dim()

    @torch.no_grad()
    def runtime_get_dim(self):
        tensor = torch.randn(1, 3, 224, 224).to(self.device)
        out = self.clip_encoder.encode_image(tensor)
        self.enc_dim = out.shape[-1] * 2 if "concat" in self.mix_mode else out.shape[-1]

    def forward(self, intermediate_representations, zero_shot_x, alpha_keys=None):
        return self.retrieval_branch(intermediate_representations, zero_shot_x, alpha_keys)
