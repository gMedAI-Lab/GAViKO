import torch
from torch import nn
from einops import repeat
import copy
from torch.nn import functional as F
import math
from utils.load_pretrained  import load_pretrain, mapping_vit
import logging
from model.vision_transformer import Attention, FeedForward


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptRelevantEstimator(nn.Module):
    def __init__(self, latent_dim, num_prompts):
        super().__init__()
        self.cls_analyzer_ = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_prompts),
            nn.Sigmoid()
        )

    @property
    def cls_analyzer(self):
        """
        Returns the CLS analyzer module.
        """
        return self.cls_analyzer_
    
    def __getitem__(self, index):
        """Make the object subscriptable"""
        return self.cls_analyzer_[index]
    
    def forward(self, cls_token):
        """
        cls_token: [B, 1, latent_dim]
        Returns: [B, 1, num_prompts] - Importance scores for each prompt
        """
        return self.cls_analyzer(cls_token)
class PromptContextFusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gl_balancer_ = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
    def __getitem__(self, index):
        """Make the object subscriptable"""
        return self.gl_balancer_[index]
    @property
    def gl_balancer(self):
        """
        Returns the Global-Local balancer module.
        """
        return self.gl_balancer_
    def forward(self, cls_token):
        """
        cls_token: [B, 1, latent_dim]
        Returns: [B, 1, 1] - Global-local balance weight
        """
        return self.gl_balancer(cls_token)

class BaseFusionAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = latent_dim ** -0.5
        
    def get_query(self, prompts_latent):
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_tokens(self, x_latent):
        raise NotImplementedError("Subclasses must implement this method")
        
    def forward(self, x_latent, prompts_latent):
        # Common attention logic
        tokens = self.get_tokens(x_latent)
        q = self.get_query(prompts_latent)
        
        # Scaled dot-product attention
        attn_weights = torch.einsum('bpd,bnd->bpn', q, tokens) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.einsum('bpn,bnd->bpd', attn_weights, tokens)
        
        return context


class GlobalAttention(BaseFusionAttention):
    def __init__(self, latent_dim, num_prompts):
        super().__init__(latent_dim)
        self.num_prompts = num_prompts
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        
    def get_query(self, prompts_latent):
        return self.query_proj(prompts_latent)
        
    def get_tokens(self, x_latent):
        return x_latent[:, self.num_prompts+1:]


class LocalAttention(BaseFusionAttention):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        
    def get_query(self, prompts_latent):
        return self.query_proj(prompts_latent)
        
    def get_tokens(self, local_latent):
        return local_latent

class Awakening_Prompt(nn.Module): # GPA
    def __init__(self, dim, num_prompts, prompt_latent_dim=20):
        super().__init__()
        self.latent_dim = prompt_latent_dim
        self.num_prompts = num_prompts
        self.scale = dim ** -0.5
        # Projection layers
        self.proj_down = nn.Sequential(
            nn.Linear(dim, self.latent_dim),
            QuickGELU()
        )
        self.proj_up = nn.Linear(self.latent_dim, dim)

        # CLS token analysis for prompt importance (Prompt Relevant Estimator)
        self.cls_analyzer = PromptRelevantEstimator(self.latent_dim, self.num_prompts)

        # Global-Local balancer (dynamic per layer) (Prompt Context Fusion)
        self.gl_balancer = PromptContextFusion(self.latent_dim)

        # Separate queries for global and local path
        self.global_attention = GlobalAttention(self.latent_dim,self.num_prompts)
        self.local_attention = LocalAttention(self.latent_dim)

        self.global_query = self.global_attention.query_proj
        self.local_query = self.local_attention.query_proj
        
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x, local_tokens):
        """
        x: [B, num_prompts + 1 + N, dim] - Global tokens (prompts + cls + image)
        local_tokens: [B, N, dim] - Local tokens (only image)
        """
        # Project to latent space
        x_latent = self.proj_down(x)
        local_latent = self.proj_down(local_tokens)

        # Extract components
        prompts_latent = x_latent[:, :self.num_prompts]  # [B, num_prompts, latent_dim]
        cls_latent = x_latent[:, self.num_prompts:self.num_prompts+1]  # [B, 1, latent_dim]
        global_img_latent = x_latent[:, self.num_prompts+1:]  # [B, N, latent_dim]

        # Analyze CLS token to determine prompt importance
        prompt_importance = self.cls_analyzer.forward(cls_latent)  # [B, 1, num_prompts]

        # Determine global-local balance from CLS token
        global_weight = self.gl_balancer.forward(cls_latent)  # [B, 1, 1]

        # GLOBAL PATH: Cross-attention between prompts and global image tokens
        global_context = self.global_attention.forward(global_img_latent, prompts_latent)  # [B, num_prompts, latent_dim]
        # LOCAL PATH: Cross-attention between prompts and local context
        local_context = self.local_attention.forward(local_latent, prompts_latent)  # [B, num_prompts, latent_dim]

        # Dynamic fusion of global and local context
        fused_prompts = global_weight * global_context + (1 - global_weight) * local_context

        # Apply prompt importance
        enhanced_prompts = fused_prompts * prompt_importance.transpose(1, 2)

        # Combine components
        combined_latent = torch.cat([
            enhanced_prompts,
            cls_latent,
            global_img_latent
        ], dim=1)

        return self.proj_up(combined_latent)

class LocalSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 local_k=(3, 6, 6),
                 DHW=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 local_dim=20,
                 qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.latent_dim=local_dim
        self.norm = nn.LayerNorm(dim)
        self.proj_down = nn.Linear(dim, self.latent_dim)
        self.qkv = nn.Linear(self.latent_dim, self.latent_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_up = nn.Linear(self.latent_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.DHW = DHW  # (D, H, W)

        if DHW is not None:
            D, H, W = DHW
            dk, hk, wk = local_k
            N = D * H * W  # 1000 (Image tokens)

            # Init mask Image Tokens (N, N)
            mask = torch.ones((N, D + dk - 1, H + hk - 1, W + wk - 1), dtype=torch.float16)
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        mask[d * H * W + h * W + w, d:d + dk, h:h + hk, w:w + wk] = 0.0

            mask_pytorch = mask[:, dk // 2:D + dk // 2, hk // 2:H + hk // 2, wk // 2:W + wk // 2].reshape(N, -1)
            mask_inf = torch.full((N, N), float('-inf'), dtype=torch.float16)
            local_mask = torch.where(mask_pytorch < 1, mask_pytorch, mask_inf)  # (N, N)
            self.mask = local_mask.unsqueeze(0)  # (1, N, N)

    def forward(self, x):
        B, N, C = x.shape  # N = num_prompt + image_tokens
        x_latent = self.proj_down(self.norm(x)) # B, N, latent_dim
        qkv = self.qkv(x_latent).chunk(3, dim=-1)  # (3, B, N, latent_dim)
        q, k, v = qkv

        attn = q @ k.transpose(-2, -1) * self.scale  # (B, N, N)
        if self.DHW is not None:
            attn = attn + self.mask.expand_as(attn).to(attn.device)  # (B, N, N) -> broadcast
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v)
        x_out = self.proj_up(x_out) # (B, N, dim)
        x_out = self.proj_drop(x_out)
        return x_out

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 num_prompts,
                 prompt_latent_dim,
                 DHW,
                 local_k,
                 share_factor=1,  # Parameter for sharing (s)
                 attn_drop=0.,
                 proj_drop=0.,
                 local_dim=20,
                 dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_prompts = num_prompts
        self.depth = depth
        self.share_factor = share_factor

        num_unique_components = math.ceil(depth / share_factor)

        self.local_attns = nn.ModuleList([
            LocalSelfAttention(dim, local_k, DHW, attn_drop, proj_drop, local_dim, qkv_bias=False)
            for _ in range(num_unique_components)
        ])

        self.prompt_projs = nn.ModuleList([
            Awakening_Prompt(dim, num_prompts, prompt_latent_dim)
            for _ in range(num_unique_components)
        ])

        self.attns = nn.ModuleList([
            Attention(dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])

        self.mlps = nn.ModuleList([
            FeedForward(dim, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, global_tokens, local_tokens):
        """
        global_tokens: [B, num_prompts + 1 + N, dim] (prompts + cls + image tokens)
        local_tokens: [B, N, dim] (only image tokens)
        """
        local_features = local_tokens

        for i in range(self.depth):
            shared_idx = i // self.share_factor

            local_features = self.local_attns[shared_idx](local_features) + local_features
            global_tokens = self.attns[i](global_tokens) + global_tokens
            prompt = self.prompt_projs[shared_idx](global_tokens, local_features)
            global_tokens = self.mlps[i](global_tokens) + global_tokens + prompt

        return self.norm(global_tokens)

class AdaptiveFusionHead(nn.Module):
    def __init__(self, dim, num_prompts, num_classes):
        super().__init__()
        self.head = nn.Linear(dim, num_classes)
        self.num_prompts = num_prompts

    def forward(self, x):
        bs, _, dim = x.shape
        out = self.head(x[:, 0:self.num_prompts+1].mean(dim=1))  # (bs, num_classes)  Prompt-CLS Pool
        # out = self.head(x[:,self.num_prompts])  # (bs, num_classes) # CLS Pool
        # out = self.head(x[:, 0:self.num_prompts].mean(dim=1)) # Prompt Pool
        # out = self.head(x[:,self.num_prompts+1:].mean(dim=1)) # Image Pool

        # prompt = x[:,0:self.num_prompts]
        # image = x[:,self.num_prompts+1:]
        # out = torch.cat([prompt, image], dim=1)
        # out = self.head(out) # Prompt-Image Pool
        return out

class Gaviko(nn.Module):
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
                 channels = 1,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 backbone=None,
                 num_prompts=8,
                 prompt_latent_dim=20,
                 local_dim=20,
                 local_k=(3,6,6),
                 DHW = (10,10,10),
                 attn_drop = 0.2,
                 proj_drop = 0.2,
                 freeze_vit=False,
                 share_factor = 1,
                 **kwargs
                ):
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
        self.num_prompts = num_prompts
        self.local_dim = local_dim
        self.local_k = local_k
        self.prompt_latent_dim = prompt_latent_dim
    
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_proj = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=(frame_patch_size, image_patch_size, image_patch_size), stride=(frame_patch_size, image_patch_size, image_patch_size),
        ))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim,
                                       depth,
                                       heads,
                                       dim_head,
                                       mlp_dim,
                                       num_prompts,
                                       prompt_latent_dim,
                                       DHW,
                                       local_k,
                                       share_factor,
                                       attn_drop,
                                       proj_drop,
                                       local_dim,
                                       dropout)


        self.pool = pool
        self.to_latent = nn.Identity()

        ##########################################
        '''
        Adaptive Fusion Head
        '''
        self.mlp_head = AdaptiveFusionHead(dim, num_prompts, num_classes)

        ##########################################
        '''
        Prompt Generator
        '''
        scale = dim ** -0.5
        self.prompt_positional_embedding = nn.Parameter(scale * torch.randn(1, num_prompts, dim))
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, dim))

        ##########################################

    
        self.freeze_vit = freeze_vit
        if self.freeze_vit:
            for k, p in self.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k): # or "overlap_proj" in k):# or "fusion_attn" in k):
                    p.requires_grad = False
                if ("head" in k or "prompt" in k or "local_attn" in k):
                    p.requires_grad = True

        if backbone is not None:
            logging.info(f'Loading pretrained {backbone}...')
            save_pretrain_dir = './pretrained'
            new_dict = load_pretrain(backbone, self.num_patches, self.conv_proj[0].weight.shape[2],save_pretrain_dir)
            self.load_state_dict(new_dict, strict=False)
            logging.info(f'Load pretrained {backbone} sucessfully!')

        self.init_weights()

    def init_weights(model, scale_factor=1.0):
        """
        initialization of the model parameters
        """
        # 1. Initialize prompt embeddings - Theo phân phối truncated normal
        with torch.no_grad():
            # Prompt embeddings theo phân phối chuẩn cắt ngọn với độ lệch chuẩn nhỏ
            # để khởi tạo feature gần 0 (neutral starting point)
            dim = model.prompt_embeddings.shape[-1]
            std = 0.02 * scale_factor
            model.prompt_embeddings.data.normal_(mean=0.0, std=std)
            model.prompt_embeddings.data = torch.clamp(model.prompt_embeddings.data, min=-2*std, max=2*std)

            # Positional embeddings với biên độ nhỏ hơn để có gradient ổn định
            pos_std = 0.01 * scale_factor
            model.prompt_positional_embedding.data.normal_(mean=0.0, std=pos_std)
            logging.info(f"Initializing prompt embeddings have {model.num_prompts} prompts ...")

        # 2. Initialize Awakening_Prompt modules
        for prompt_proj in model.transformer.prompt_projs:

            # Projection layers - Xavier/Glorot initialization với gain nhỏ để gradient flows tốt hơn
            nn.init.xavier_uniform_(prompt_proj.proj_down[0].weight, gain=0.7*scale_factor)
            nn.init.zeros_(prompt_proj.proj_down[0].bias)
            nn.init.xavier_uniform_(prompt_proj.proj_up.weight, gain=0.7*scale_factor)
            nn.init.zeros_(prompt_proj.proj_up.bias)

            # Query projections - Orthogonal initialization for attention
            nn.init.orthogonal_(prompt_proj.global_query.weight, gain=scale_factor)
            nn.init.zeros_(prompt_proj.global_query.bias)
            nn.init.orthogonal_(prompt_proj.local_query.weight, gain=scale_factor)
            nn.init.zeros_(prompt_proj.local_query.bias)

            # CLS analyzer - Xavier initialization với bias=0
            # Sigmoid layer cuối có bias lớn hơn 0 để bắt đầu với giá trị gần 0.5
            nn.init.xavier_uniform_(prompt_proj.cls_analyzer[1].weight, gain=1.0)
            nn.init.zeros_(prompt_proj.cls_analyzer[1].bias)
            nn.init.xavier_uniform_(prompt_proj.cls_analyzer[3].weight, gain=1.0)
            nn.init.constant_(prompt_proj.cls_analyzer[3].bias, 0.0)  # Sigmoid(0) = 0.5

            # Global-Local balancer - Bias to start slightly global-dominated
            nn.init.xavier_uniform_(prompt_proj.gl_balancer[1].weight, gain=1.0)
            nn.init.constant_(prompt_proj.gl_balancer[1].bias, 0.5)  # Bias để bắt đầu với global:0.62, local:0.38
        logging.info(f"Initializing Prompt Awakener with prompt latent dim {model.prompt_latent_dim}...")

        # 3. Initialize LocalSelfAttention layers
        for local_attn in model.transformer.local_attns:
            # Input Projection
            nn.init.xavier_uniform_(local_attn.proj_down.weight, gain=0.5*scale_factor)  # Smaller gain for stable training
            nn.init.zeros_(local_attn.proj_down.bias)

            # QKV projection - trọng số QKV được scale để tránh gradient vanishing
            nn.init.xavier_uniform_(local_attn.qkv.weight, gain=1.0)
            if local_attn.qkv.bias is not None:
                nn.init.zeros_(local_attn.qkv.bias)

            # Output projection
            nn.init.xavier_uniform_(local_attn.proj_up.weight, gain=0.5*scale_factor)  # Smaller gain for stable training
            nn.init.zeros_(local_attn.proj_down.bias)
        logging.info(f"Initializing local context Extractor with Local Window size {model.local_k} and Local latent dimension {model.local_dim} ...")


        # 4. MLP Head
        nn.init.xavier_uniform_(model.mlp_head.head.weight)
        nn.init.zeros_(model.mlp_head.head.bias)
        logging.info("Initializing Linear head ...")
        logging.info("Initialization completed successfully!")

    def train(self, mode=True):
        if mode:
            # Ensure ViT encoder stays in eval mode if frozen
            super().train(mode)
            if self.freeze_vit:
                self.transformer.eval()
                self.conv_proj.eval()
                self.dropout.eval()

                self.transformer.local_attns.train()
                self.transformer.prompt_projs.train()
                self.mlp_head.train()
        else:
            # Set all submodules to evaluation mode
            for module in self.children():
                module.eval()


    def forward(self, img):
        x = self.conv_proj(img)
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        global_tokens = torch.cat((cls_tokens, x), dim=1)

        ###################
        global_tokens = torch.cat([self.prompt_embeddings.expand(global_tokens.shape[0], -1, -1), global_tokens], dim=1) # bs, num_prompts + N, 768
        global_tokens = global_tokens + torch.cat([
                            self.prompt_positional_embedding.to(global_tokens.dtype),
                            self.pos_embedding.to(global_tokens.dtype)], dim=1)
        global_tokens = self.dropout(global_tokens)
        ###################
        local_tokens = x.clone()
        local_tokens = local_tokens + self.pos_embedding[:,1:,:].to(local_tokens.dtype)
        local_tokens = self.dropout(local_tokens)

        out = self.transformer(global_tokens, local_tokens)
        return self.mlp_head(out)