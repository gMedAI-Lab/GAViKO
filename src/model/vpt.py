# This implementation is adapted from:
# https://github.com/KMnP/vpt/blob/main/src/models/vit_prompt/vit.py
# Original author: Menglin Jia

import torch
from torch import nn

from einops import rearrange, repeat
import logging
# helpers
from model.vision_transformer import VisionTransformer
from utils.load_pretrained  import mapping_vit


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
    
class PromptedVisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 image_patch_size: int,
                 frames: int,
                 frame_patch_size : int,
                #  num_layers: int,
                #  num_heads: int,
                #  hidden_dim: int,
                #  mlp_dim: int,
                 dropout: 0.0,
                 emb_dropout: 0.0,
                 num_classes = 5,
                 channels = 3,
                 dim_head = 64,
                 freeze_vit = True,
                 pool = 'cls',
                 backbone = None,
                 prompt_dropout = 0.0,
                 prompt_dim = 64,
                 num_prompts = 8,
                 deep_prompt = True,
                 **kwargs
                 ):
        super().__init__()
        num_layers, num_heads, hidden_dim, mlp_dim = mapping_vit(backbone)

        self.image_size = image_size
        self.num_layers = num_layers
        self.image_patch_size = image_patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.deep_prompt = deep_prompt
        with open('deep_prompt.txt', 'a') as f:
            f.write(f'Deep prompt: {self.deep_prompt}\n')
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        if self.deep_prompt:
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    num_layers, num_prompts, prompt_dim))
            # xavier_uniform initialization
            nn.init.xavier_uniform_(self.deep_prompt_embeddings.data)
        else:
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_prompts, prompt_dim))
             # xavier_uniform initialization
            nn.init.xavier_uniform_(self.prompt_embeddings.data)

        self.vision_transformer = VisionTransformer(
                                     image_size=image_size,
                                     image_patch_size=image_patch_size,
                                     frames=frames,
                                     frame_patch_size=frame_patch_size,
                                     num_classes=num_classes,
                                     dim=hidden_dim,
                                     depth=num_layers,
                                     heads=num_heads,
                                     mlp_dim=mlp_dim,
                                     pool = pool,
                                     channels = channels,
                                     dim_head = dim_head,
                                     dropout = dropout,
                                     emb_dropout = emb_dropout,
                                     backbone=backbone)
        self.freeze_vit = freeze_vit

        self.init_head_weights()
        self.init_promptproj_weights()

        if self.freeze_vit:
            for k, p in self.vision_transformer.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k):
                    p.requires_grad = False

    def init_head_weights(self):
        nn.init.xavier_uniform_(self.vision_transformer.mlp_head.weight)
        nn.init.zeros_(self.vision_transformer.mlp_head.bias)
        logging.info("Initialize head weight successfully!")

    def init_promptproj_weights(self):
        nn.init.xavier_uniform_(self.prompt_proj.weight)
        nn.init.zeros_(self.prompt_proj.bias)
        logging.info("Initialize prompt projector successfully!")

    def train(self, mode=True):
        if mode:
            # Ensure ViT encoder stays in eval mode if frozen
            super().train(mode)
            if self.freeze_vit:
                self.vision_transformer.transformer.eval()
                self.vision_transformer.conv_proj.eval()
                self.vision_transformer.dropout.eval()
                self.vision_transformer.mlp_head.train()
                self.prompt_proj.train()
        else:
            # Set all submodules to evaluation mode
            for module in self.children():
                module.eval()

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_transformer.conv_proj(x)

    def forward_prompt(self, x):
        B = x.shape[0] # (batch_size, 1 + n_patches, hidden_dim)
        # after CLS token, all before image patches
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        for i in range(self.num_layers):
            attn, ff = self.vision_transformer.transformer.layers[i]
            x = attn(x) + x
            x = ff(x) + x
        x = self.vision_transformer.transformer.norm(x)
        x = x.mean(dim=1) if self.vision_transformer.pool == 'mean' else x[:, 0]
        x = self.vision_transformer.to_latent(x)
        return self.vision_transformer.mlp_head(x)

    def forward_deep_prompt(self, x):
        num_layers = self.num_layers
        B = x.shape[0]
        for i in range(num_layers):
            if i == 0:
                x = torch.cat((x[:,:1,:],
                               self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i].expand(B, -1, -1))),
                               x[:,1:,:]), dim=1) # Combine prompt embeddings with image embeddings (after cls tokens before patch embeddings) at the first Transformer layer
            else:
                x = torch.cat((x[:,:1,:],
                               self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i].expand(B, -1, -1))),
                               x[:,(1 + self.deep_prompt_embeddings[i].shape[1]):,:]), dim=1) # after cls tokens & prompt embeddings at the current layers (to avoid overiding prompt embedding in previous, help keep info from current layers) before patch embeddings

            attn, ff = self.vision_transformer.transformer.layers[i]
            x = attn(x) + x
            x = ff(x) + x
        x = self.vision_transformer.transformer.norm(x)
        x = x.mean(dim=1) if self.vision_transformer.pool == 'mean' else x[:, 0]
        x = self.vision_transformer.to_latent(x)
        return self.vision_transformer.mlp_head(x)


    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape

        cls_tokens = repeat(self.vision_transformer.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vision_transformer.pos_embedding[:, :(n + 1)]
        x = self.vision_transformer.dropout(x)

        if self.deep_prompt==True:
            return self.forward_deep_prompt(x)
        else:
            return self.forward_prompt(x)