import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
# helpers

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
                 dim, depth,
                 heads,
                 mlp_dim,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 pretrain_path=None):
        super().__init__()
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

        if pretrain_path is not None:
            self.load_pretrain(pretrain_path)
            print(f'Load pretrained {pretrain_path} sucessfully!')

    def load_pretrain(self, backbone):
        import timm
        import os

        save_dir = './pretrained'
        os.makedirs(save_dir, exist_ok=True)

        if backbone.lower() == 'vit-b16':
            backbone_type = 'vit_base_patch16_224_in21k'
        elif backbone.lower() == 'vit-b32':
            backbone_type = 'vit_tiny_patch16_224_in21k'
        elif backbone.lower() == 'vit-s16':
            backbone_type = 'vit_small_patch16_224_in21k'
        elif backbone.lower() == 'vit-l16':
            backbone_type = 'vit_large_patch16_224_in21k'
        else:
            print('Warning: The model initizalizes without pretrained knowledge!')
        model = timm.create_model(backbone_type, pretrained=True)

        # Lưu state_dict
        save_path = os.path.join(save_dir, backbone)
        torch.save(model.state_dict(), save_path)

        print(f"Pretrained {backbone} download successfully!'")
        jax_dict = torch.load(save_path, map_location='cpu')

        new_dict = {}

        def interpolate_pos_embedding(pre_pos_embed):
            cls_token, pretrained_pos_embed = pre_pos_embed[:, :1, :], pre_pos_embed[:, 1:, :]  # [1, 1, 768], [1, 196, 768]
            new_num_patches = self.num_patches # 1000
            old_num_patches = int(pretrained_pos_embed.shape[1] ** 0.5) # 14
            pretrained_pos_embed = pretrained_pos_embed.reshape(1, old_num_patches, old_num_patches, -1).permute(0, 3, 1, 2)  # [1, 768, 14, 14]
            pretrained_pos_embed = pretrained_pos_embed.unsqueeze(2)  # [1, 768, 1, 14, 14]
            new_size = round(new_num_patches ** (1/3))
            pretrained_pos_embed = F.interpolate(pretrained_pos_embed, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)  # [1, 768, 10, 10, 10]
            pretrained_pos_embed = pretrained_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, new_size*new_size*new_size, -1) # [1,1000, 768]
            new_pos_embed = torch.cat([cls_token, pretrained_pos_embed], dim=1)
            return new_pos_embed

        def mean_kernel(patch_emb_weight):
            patch_emb_weight = patch_emb_weight.mean(dim=1, keepdim=True)  # Shape: [768, 1, 16, 16]
            depth = self.conv_proj[0].weight.shape[2]
            patch_emb_weight = patch_emb_weight.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # Shape: [768, 1, 12, 16, 16]
            return patch_emb_weight

        def add_item(key, value):
            key = key.replace('blocks', 'transformer.layers')
            new_dict[key] = value

        for key, value in jax_dict.items():
            if key == 'cls_token':
                new_dict[key] = value

            elif 'norm1' in key:
                new_key = key.replace('norm1', '0.norm')
                add_item(new_key, value)
            elif 'attn.qkv' in key:
                new_key = key.replace('attn.qkv', '0.to_qkv')
                add_item(new_key, value)
            elif 'attn.proj' in key:
                new_key = key.replace('attn.proj', '0.to_out.0')
                add_item(new_key, value)
            elif 'norm2' in key:
                new_key = key.replace('norm2', '1.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '1.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '1.net.4')
                add_item(new_key, value)
            elif 'patch_embed.proj.weight' in key:
                new_key = key.replace('patch_embed.proj.weight', 'conv_proj.0.weight')
                value = mean_kernel(value)
                add_item(new_key, value)
            elif 'patch_embed.proj.bias' in key:
                new_key = key.replace('patch_embed.proj.bias', 'conv_proj.0.bias')
                add_item(new_key, value)
            elif key == 'pos_embed':
                value = interpolate_pos_embedding(value)
                add_item('pos_embedding', value)
            elif key == 'norm.weight':
                add_item('transformer.norm.weight', value)
            elif key == 'norm.bias':
                add_item('transformer.norm.bias', value)

        self.load_state_dict(new_dict, strict=False)


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
    
class PromptedVisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 image_patch_size: int,
                 frames: int,
                 frame_patch_size : int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: 0.0,
                 emb_dropout: 0.0,
                 num_classes = 5,
                 channels = 3,
                 dim_head = 64,
                 freeze_vit = True,
                 pool = 'cls',
                 pretrain_path = None,
                 prompt_dropout = 0.0,
                 prompt_dim = 64,
                 num_prompts = 8,
                 deep_prompt = True,
                 ):
        super().__init__()
        self.image_size = image_size
        self.num_layers = num_layers
        self.image_patch_size = image_patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.deep_prompt = deep_prompt

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
                                     pretrain_path=pretrain_path)
        self.freeze_vit = freeze_vit

        # self.init_head_weights()
        self.init_promptproj_weights()

        if self.freeze_vit:
            for k, p in self.vision_transformer.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k):
                    p.requires_grad = False

    def init_head_weights(self):
        nn.init.xavier_uniform_(self.vision_transformer.mlp_head.weight)
        nn.init.zeros_(self.vision_transformer.mlp_head.bias)
        print("Initialize head weight successfully!")

    def init_promptproj_weights(self):
        nn.init.xavier_uniform_(self.prompt_proj.weight)
        nn.init.zeros_(self.prompt_proj.bias)
        print("Initialize prompt projector successfully!")

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