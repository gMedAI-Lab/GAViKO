# This implementation is adapted from:
# https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# Original author: James Q. Freeman
# License: GPL-3.0

import torch
from torch import nn
import math

class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv
    
class MeLO(nn.Module):
    def __init__(self, vit, r:int, alpha:int, num_classes:int, lora_layer=None, **kwargs):
        super(MeLO, self).__init__()
        assert r > 0
        assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit.transformer.layers)))
        # self.dim = vit.mlp_head.in_features # 768
        self.w_As = []
        self.w_Bs = []

        # Freeze ViT
        for param in vit.parameters():
            param.requires_grad = False

        for t_layer_i, (attn, mlp) in enumerate(vit.transformer.layers):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = attn.to_qkv
            self.dim = w_qkv_linear.in_features # 768
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            attn.to_qkv =  _LoRA_qkv_timm(
                                w_qkv_linear,
                                w_a_linear_q,
                                w_b_linear_q,
                                w_a_linear_v,
                                w_b_linear_v,
                                r,
                                alpha
                            )
        self.reset_parameters()
        self.lora_vit = vit
        if num_classes > 0:
            self.lora_vit.mlp_head = nn.Linear(self.dim, num_classes)


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        return self.lora_vit(x)