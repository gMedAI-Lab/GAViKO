# This implementation is adapted from:
# https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py
# Original author: Shoufa Chen
# License: MIT

import torch
from torch import nn

from einops import rearrange, repeat
import math
import logging

# helpers
import model.vision_transformer as vision_transformer
from utils.load_pretrained  import load_pretrain, mapping_vit

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Adapter(nn.Module):
    def __init__(self,
                 d_dim,
                 down_dim=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.d_dim = d_dim
        self.down_dim = down_dim

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.d_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_adapter_proj = nn.Linear(self.d_dim, self.down_dim)
        self.non_linear_func = nn.ReLU()
        self.up_adapter_proj = nn.Linear(self.down_dim, self.d_dim)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_adapter_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_adapter_proj.weight)
                nn.init.zeros_(self.down_adapter_proj.bias)
                nn.init.zeros_(self.up_adapter_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_adapter_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout)
        up = self.up_adapter_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                vision_transformer.Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,),
                Adapter(dim),
                vision_transformer.FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, adapter, ff in self.layers:
            x = attn(x) + x
            residual = adapter(x)
            x = ff(x) + x + residual

        return self.norm(x)

class AdaptFormer(nn.Module):
    def __init__(self,
                 *,
                 image_size,
                 image_patch_size,
                 frames,
                 frame_patch_size,
                 num_classes,
                #  dim, depth,
                #  heads,
                #  mlp_dim,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 backbone=None,
                 freeze_vit=False,
                 **kwargs):
        super().__init__()
        depth, heads, dim, mlp_dim = mapping_vit(backbone)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        self.num_patches = num_patches
        self.image_size = image_size
        self.image_patch_size = image_patch_size
        self.frames = frames
        self.frame_patch_size=frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_proj = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=(frame_patch_size, image_patch_size, image_patch_size), stride=(frame_patch_size, image_patch_size, image_patch_size),
        ))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        if backbone is not None:
            logging.info(f'Loading pretrained {backbone}...')
            save_pretrain_dir = './pretrained'
            new_dict = load_pretrain(backbone, self.num_patches, self.conv_proj[0].weight.shape[2],save_pretrain_dir)
            self.load_state_dict(new_dict, strict=False)
            logging.info(f'Load pretrained {backbone} sucessfully!')

        self.freeze_vit = freeze_vit

        self.init_head_weights()

        if self.freeze_vit:
            for k, p in self.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k):
                    p.requires_grad = False
                if "adapter" in k or "head" in k:
                    p.requires_grad = True

    def init_head_weights(self):
        nn.init.xavier_uniform_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)
        logging.info("Initialize head weight successfully!")

    def train(self, mode=True):
        if mode:
            # Ensure ViT encoder stays in eval mode if frozen
            super().train(mode)
            if self.freeze_vit:
                self.transformer.eval()
                self.conv_proj.eval()
                self.dropout.eval()
                self.mlp_head.train()

                for layer in self.transformer.layers:
                    layer[1].train()
        else:
            # Set all submodules to evaluation mode
            for module in self.children():
                module.eval()


    def forward(self, img):
        x = self.conv_proj(img)
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)