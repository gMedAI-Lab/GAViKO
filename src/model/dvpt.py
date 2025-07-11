import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import math
import logging
# helpers
import model.transformer_vanilla as transformer_vanilla
from utils.load_pretrained  import load_pretrain , mapping_vit

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

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class share_MLP(nn.Module):
    def __init__(self, d_model, num_prompts):
        super().__init__()
        self.latent_dim=20
        self.prompt_key_proj_d = nn.Linear(d_model, self.latent_dim)
        self.prompt_key_proj_u = nn.Linear(self.latent_dim, d_model)
        self.prompt_gate = torch.nn.Parameter(torch.zeros(1))
        self.gellu = QuickGELU()

        self.softmax = nn.Softmax(dim=-1)
        self.num = num_prompts
        self.scale = d_model ** -0.5

    def forward(self, x):
        x = self.prompt_key_proj_d(self.gellu(x)) # bs, num_prompt + N, 20
        cls_token = x[:,self.num:self.num+1,:]
        prompt = x[:,0:self.num,:] # bs, num_prompt_tokens, 20
        tokens = x[:, self.num+1:,:] # bs, N, 20
        prompt_attn = (prompt@tokens.transpose(-2,-1)*self.scale) # bs, num_prompt, N
        prompt_attn = self.softmax(prompt_attn)
        prompt_out = (prompt_attn@tokens) # bs, num_prompt, 20
        prompt_out = torch.cat([prompt_out, cls_token, tokens], dim=1) # bs, num_prompt + N, 20
        prompt_out = self.prompt_key_proj_u(prompt_out) * self.prompt_gate # bs, num_prompt + N, 768
        return prompt_out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, num_prompts, dropout):
        super().__init__()

        self.attn = transformer_vanilla.Attention(dim, heads, dim_head, dropout)
        self.mlp = transformer_vanilla.FeedForward(dim, mlp_dim, dropout)
        self.prompt_proj = share_MLP(dim, num_prompts)


    def forward(self, x: torch.Tensor):
        x = self.attn(x) + x
        prompt = self.prompt_proj(x)
        x = self.mlp(x) + x + prompt
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_prompts, dropout = 0., pool='cls'):
        super().__init__()
        self.num = num_prompts
        self.norm = nn.LayerNorm(dim)
        self.pool = pool
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualAttentionBlock(dim, heads, dim_head, mlp_dim, num_prompts, dropout)
            ]))

    def forward(self, x):
        for rab in self.layers:
            x = rab[0](x)
        if self.pool=='cls':
            return self.norm(x)
        else:
            return self.norm(x[:, 0:self.num+1, :])

class DynamicVisualPromptTuning(nn.Module):
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
                 num_prompts = 50,
                 freeze_vit=False,
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

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        ##########################
        scale = dim ** -0.5
        self.prompt_positional_embedding = nn.Parameter(scale * torch.randn(1, num_prompts, dim))
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, dim))
        ##########################

        self.conv_proj = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=(frame_patch_size, image_patch_size, image_patch_size), stride=(frame_patch_size, image_patch_size, image_patch_size),
        ))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_prompts, dropout, pool)

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
                if "prompt" in k or "head" in k:
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
                    layer[0].prompt_proj.train()
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

        ###################
        x = torch.cat([self.prompt_embeddings.expand(x.shape[0], -1, -1), x], dim=1) # bs, num_prompts + N, 768
        x = x + torch.cat([self.prompt_positional_embedding.to(x.dtype), self.pos_embedding.to(x.dtype)], dim=1)
        ###################

        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)