import torch
from .base_network import BaseNetwork
import clip
import torch.nn as nn

class CLIPModule(BaseNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        # Load the CLIP model and set it to evaluation mode
        self.clip_encoder = (
            clip.load(args.backbone, jit=False, device=self.device)[0]
            .eval()
            .requires_grad_(False)
        )
        self.clip_encoder = self.clip_encoder.float()

    def forward(self, x):
        # Encode the input image using CLIP's image encoder
        return self.clip_encoder.encode_image(x)

class LearnableCLIPModule(BaseNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        # Load the CLIP model and set it to evaluation mode
        self.clip_encoder = (
            clip.load(args.backbone, jit=False, device=self.device)[0]
            .eval()
            .requires_grad_(False)
        )
        self.clip_encoder = self.clip_encoder.float()
        # Split the transformer into fixed and trainable parts
        self.transformer_fix_part = self.clip_encoder.visual.transformer.resblocks[
            : args.train_transformer_block_index
        ]
        self.transformer_trainable_part = (
            self.clip_encoder.visual.transformer.resblocks[
                args.train_transformer_block_index:
            ]
        )

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument(
            "--train_transformer_block_index",
            type=int,
            default=11,
        )
        parser.parse_known_args()
        return parser

    def image_encoder_fix_part_forward(self, x):
        # Forward pass through the fixed part of the image encoder
        x = self.clip_encoder.visual.conv1(x.type(self.clip_encoder.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.clip_encoder.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.clip_encoder.visual.positional_embedding.to(x.dtype)
        x = self.clip_encoder.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_fix_part(x)
        return x

    def image_encoder_trainable_part_forward(self, x):
        # Forward pass through the trainable part of the image encoder
        
        x = self.transformer_trainable_part(x)
        x = x.permute(1, 0, 2)

        x = self.clip_encoder.visual.ln_post(x[:, 0, :])

        if self.clip_encoder.visual.proj is not None:
            x = x @ self.clip_encoder.visual.proj

        return x

    def forward(self, x):
        # Full forward pass through both fixed and trainable parts
        x = self.image_encoder_fix_part_forward(x)
        x = self.image_encoder_trainable_part_forward(x)
        return x
    
    def encode_image(self, x):
        return self.clip_encoder.encode_image(x)
    
    def encode_text(self, x):
        return self.clip_encoder.encode_text(x)


class ModifiedResidualAttentionBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def attention(self, x, key, value):
        # Apply attention mechanism
        self.block.attn_mask = self.block.attn_mask.to(dtype=x.dtype, device=x.device) if self.block.attn_mask is not None else None
        return self.block.attn(x, key, value, need_weights=False, attn_mask=self.block.attn_mask)[0]

    def forward(self, x, key, value):
        # Forward pass through the modified residual attention block
        x = x + self.attention(self.block.ln_1(x), self.block.ln_1(key), self.block.ln_1(value))
        x = x + self.block.mlp(self.block.ln_2(x))
        return x

class CLIPTrainablePart(BaseNetwork):
    def __init__(self, args, block, ln, proj):
        super().__init__(args)
        self.device = args.device
        self.block = block
        self.ln = ln
        self.proj = proj

        # Set up classifier or bias for "other" class
        self.other_classifier = nn.Linear(512, 1) if args.include_the_other_class and args.use_other_classifier else None
        self.other_bias = nn.Parameter(torch.zeros(1)) if args.include_the_other_class else None

    def forward(self, x):
        x = self.block(x)
        x = x.permute(1, 0, 2)
        x = self.ln(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward_with_other_logit(self, x, text_features):
        # Forward pass with additional logit for "other" class
        x = self.forward(x)
        x = x / x.norm(dim=-1, keepdim=True)
        similarity = 100 * x @ text_features.T
        other_logit = torch.nn.functional.linear(x, torch.zeros(1, x.shape[1], device=self.device), self.other_bias)
        similarity = torch.cat([similarity, other_logit], dim=1)
        return similarity

class IncrementalClassifier(nn.Module):
    def __init__(self, in_features, initial_out_features=2):
        super().__init__()
        self.classifier = torch.nn.Linear(in_features, initial_out_features)

    @torch.no_grad()
    def adaptation(self, class_features):
        # Adapt the classifier to new classes
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(self.classifier.out_features, class_features.shape[0])
        classifier = torch.nn.Linear(in_features, new_nclasses).to(self.classifier.weight.device)
        classifier.weight[:old_nclasses] = self.classifier.weight
        classifier.weight[old_nclasses:] = class_features[old_nclasses:]
        classifier.bias[:old_nclasses] = self.classifier.bias
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(x)

class CLIPTrainablePartPlusIncrementalClassifier(BaseNetwork):
    def __init__(self, args, block, ln, proj, enc_dim):
        super().__init__(args)
        self.device = args.device
        self.block = block
        self.ln = ln
        self.proj = proj
        self.other_bias = nn.Parameter(torch.zeros(1))
        self.incremental_classifier = IncrementalClassifier(in_features=enc_dim)
        self.class_features = None

    def adaptation(self, class_features):
        # Adapt the incremental classifier to new classes
        self.incremental_classifier.adaptation(class_features)
        self.class_features = class_features

    def forward(self, x):
        # Forward pass through the trainable part and incremental classifier
        x = self.block(x)
        x = x.permute(1, 0, 2)
        x = self.ln(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        x = self.incremental_classifier(x)
        return x

    def forward_with_other_logit(self, x, text_features):
        # Forward pass with additional logit for "other" class
        x = self.block(x)
        x = x.permute(1, 0, 2)
        x = self.ln(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        x = x / x.norm(dim=-1, keepdim=True)
        similarity = 100 * x @ text_features.T
        zero_weights = torch.zeros(1, x.shape[1]).to(self.device)
        zero_weights.requires_grad = False
        other_logit = torch.nn.functional.linear(x, zero_weights, self.other_bias)
        similarity = torch.cat([similarity, other_logit], dim=1)
        return similarity
