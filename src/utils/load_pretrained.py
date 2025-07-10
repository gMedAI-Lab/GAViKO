import torch
import torch.nn.functional as F
from torch import nn
import timm
import os

def load_pretrain( backbone,num_patches,depth_dim,save_dir):

    backbone = backbone.replace('_', '-')
    # save_dir = './pretrained'
    os.makedirs(save_dir, exist_ok=True)

    if backbone.lower() == 'vit-b16':
        backbone_type = 'vit_base_patch16_224_in21k'
    elif backbone.lower() == 'vit-t16':
        backbone_type = 'vit_tiny_patch16_224_in21k'
    elif backbone.lower() == 'vit-s16':
        backbone_type = 'vit_small_patch16_224_in21k'
    elif backbone.lower() == 'vit-l16':
        backbone_type = 'vit_large_patch16_224_in21k'
    else:
        print('Warning: The model initizalizes without pretrained knowledge!')
    model = timm.create_model(backbone_type, pretrained=True)

    # Lưu state_dict
    save_path = os.path.join(save_dir, backbone_type)
    torch.save(model.state_dict(), save_path)

    print(f"Pretrained {backbone} downloaded successfully")
    jax_dict = torch.load(save_path, map_location='cpu')
    new_dict = {}

    def interpolate_pos_embedding(pre_pos_embed):
        cls_token, pretrained_pos_embed = pre_pos_embed[:, :1, :], pre_pos_embed[:, 1:, :]  # [1, 1, 768], [1, 196, 768]
        new_num_patches = num_patches # 1000
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
        # depth = self.conv_proj[0].weight.shape[2]
        depth = depth_dim
        patch_emb_weight = patch_emb_weight.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # Shape: [768, 1, 12, 16, 16]
        return patch_emb_weight
    
    def add_item(key, value):
        key = key.replace('blocks', 'transformer')
        new_dict[key] = value
    def add_attn_item(key, value):
        key = key.replace('blocks', 'transformer.attns')
        new_dict[key] = value
    def add_mlp_item(key, value):
        key = key.replace('blocks', 'transformer.mlps')
        new_dict[key] = value

    for key, value in jax_dict.items():
        if key == 'cls_token':
            new_dict[key] = value

        elif 'norm1' in key:
            new_key = key.replace('norm1', 'norm')
            add_attn_item(new_key, value)
        elif 'attn.qkv' in key:
            new_key = key.replace('attn.qkv', 'to_qkv')
            add_attn_item(new_key, value)
        elif 'attn.proj' in key:
            new_key = key.replace('attn.proj', 'to_out.0')
            add_attn_item(new_key, value)
        elif 'norm2' in key:
            new_key = key.replace('norm2', 'net.0')
            add_mlp_item(new_key, value)
        elif 'mlp.fc1' in key:
            new_key = key.replace('mlp.fc1', 'net.1')
            add_mlp_item(new_key, value)
        elif 'mlp.fc2' in key:
            new_key = key.replace('mlp.fc2', 'net.4')
            add_mlp_item(new_key, value)
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
    return new_dict
    # self.load_state_dict(new_dict, strict=False)


def mapping_vit(backbone):
    vit_config_map={
        'vit-b16': {'depth': 12, 'heads': 12, 'dim': 768, 'mlp_dim': 3072},
        'vit-t16': {'depth': 12, 'heads': 3, 'dim': 192, 'mlp_dim': 768},
        'vit-s16': {'depth': 12, 'heads': 6, 'dim': 384, 'mlp_dim': 1536},
        'vit-l16': {'depth': 24, 'heads': 16, 'dim': 1024, 'mlp_dim': 4096},    
    }
    
    if backbone is not None:
        if backbone.lower() not in vit_config_map:
            raise ValueError(f"Unsupported backbone: {backbone}. Supported backbones are: {list(vit_config_map.keys())}")
        depth = vit_config_map[backbone.lower()]['depth']
        heads = vit_config_map[backbone.lower()]['heads']
        dim = vit_config_map[backbone.lower()]['dim']
        mlp_dim = vit_config_map[backbone.lower()]['mlp_dim']
        return depth, heads, dim, mlp_dim
    else:
        raise ValueError("Backbone must be specified.")
    


def load_vanilla_pretrain(backbone,config):
    def loading_vit(config):
        def pair(t):
            return t if isinstance(t, tuple) else (t, t)

        depth, heads, dim, mlp_dim = mapping_vit(backbone)
        frame_patch_size = config['model']['frame_patch_size']
        image_patch_size = config['model']['image_patch_size']
        image_size = config['model']['image_size']
        frames = config['model']['frames']
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        conv_proj = nn.Sequential(
            nn.Conv3d(config['model']['channels'], dim, kernel_size=(frame_patch_size, image_patch_size, image_patch_size), stride=(frame_patch_size, image_patch_size, image_patch_size),
        ))
        return conv_proj, num_patches

    conv_proj, num_patches = loading_vit(config)
    vannilla_pretrain_dicts = load_pretrain(backbone, num_patches, conv_proj[0].weight.shape[2], save_dir='./pretrained')
    return vannilla_pretrain_dicts

def load_vanilla_pretrain_with_adapters(backbone, config, checkpoint_path):
    vannilla_pretrain_dicts = load_vanilla_pretrain(backbone, config)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_dict = {**vannilla_pretrain_dicts, **checkpoint}
    # Filter out the keys that are not in the vannilla_pretrain_dicts
    return new_dict
