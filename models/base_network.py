from typing import List
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip_encoder = None

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def encode_class_features(self, label_texts: List[str]):
        if not label_texts:
            return None

        def encode_batch(texts):
            with torch.no_grad():
                labels = clip.tokenize(texts).to(self.device)
                features = self.clip_encoder.encode_text(labels)
            return F.normalize(features, dim=-1)

        if len(label_texts) > self.args.batch_size:
            all_features = []
            for i in range(0, len(label_texts), self.args.batch_size):
                batch = label_texts[i:i + self.args.batch_size]
                all_features.append(encode_batch(batch))
            return torch.cat(all_features, dim=0)
        else:
            return encode_batch(label_texts)
