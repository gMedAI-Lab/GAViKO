import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import math
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
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,),
                Adapter(dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
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
                 dim, depth,
                 heads,
                 mlp_dim,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 pretrain_path=None,
                 freeze_vit=False,):
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

        self.freeze_vit = freeze_vit

        # self.init_head_weights()

        if self.freeze_vit:
            for k, p in self.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k):
                    p.requires_grad = False
                if "adapter" in k or "head" in k:
                    p.requires_grad = True

    def init_head_weights(self):
        nn.init.xavier_uniform_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)
        print("Initialize head weight successfully!")

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

    def load_pretrain(self, pretrain_path):
        jax_dict = torch.load(pretrain_path, map_location='cpu')
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
                new_key = key.replace('norm2', '2.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '2.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '2.net.4')
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