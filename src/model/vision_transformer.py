# This implementation is adapted from:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
# Original author: Phil Wang (lucidrains)
# License: MIT

# Modifications in this version:
# - Replaced the original linear patch embedding with a 3D convolutional layer
# - Refactored for medical 3D MRI input shape
# - Added support for pretrained weights loading

import torch
from torch import nn

from einops import rearrange, repeat
from torch.nn import functional as F
import math
import logging
# helpers
from utils.load_pretrained  import load_pretrain, mapping_vit

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(self,
                 *,
                 image_size,
                 image_patch_size,
                 frames,
                 frame_patch_size,
                 num_classes,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 backbone=None,
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
        self.depth = depth

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