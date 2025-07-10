#print model structure and pretained ones
import os

import torch
import torch.nn as nn
from model import transformer_vanilla
from utils.load_pretrained import load_pretrain, mapping_vit
from model.gaviko import Gaviko
from model.adaptformer import AdaptFormer
from model.bifit import BiFit
from model.dvpt import DynamicVisualPromptTuning
from model.evp import ExplicitVisualPrompting
from model.ssf import ScalingShiftingFeatures
from model.melo import MedicalLoRA
from model.vpt import PromptedVisionTransformer
from omegaconf import OmegaConf
model_path = '/mnt/e/workspace/gsoft/experiments/gaviko/gaviko_vit_b16_best_model_epoch1_acc0.1667.pt'
config_path = '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_gaviko.yaml'


# list_test = ['gaviko', 'adaptformer', 'bifit', 'dvpt', 'evp', 'ssf', 'melo', 'deep_vpt', 'shallow_vpt']
list_test = ['ssf']
map_config = {
    'gaviko': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_gaviko.yaml',
    'adaptformer': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_adaptformer.yaml',
    'bifit': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_bifit.yaml',
    'dvpt': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_dvpt.yaml',
    'evp': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_evp.yaml',
    'ssf': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_ssf.yaml',
    'melo': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_melo.yaml',
    'deep_vpt': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_vpt.yaml',
    'shallow_vpt': '/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_vpt.yaml',
}

for method in list_test:
    config = OmegaConf.load(map_config[method])

    config['model']['method'] = method
    if config['model']['method'] == 'gaviko':
        model = Gaviko(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # heads=config['model']['heads'],
            # depth= config['model']['depth'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            attn_drop=config['model']['attn_drop'],
            proj_drop=config['model']['proj_drop'],
            channels=config['model']['channels'],
            num_classes=config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
            num_prompts=config['model']['num_prompts'],
            prompt_latent_dim=config['model']['prompt_latent_dim'],
            local_dim=config['model']['local_dim'],
            local_k= tuple(config['model']['local_k']),
            DHW=tuple(config['model']['DHW']),
            share_factor=config['model']['share_factor']
        )

    elif config['model']['method'] == 'adaptformer':
        model = AdaptFormer(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads=config['model']['heads'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
        )# .to(device)

    elif config['model']['method'] == 'bifit':
        model = BiFit(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads=config['model']['heads'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
        )
        for key, value in model.named_parameters():
            if "bias" in key:
                value.requires_grad = True
            elif "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif config['model']['method'] == 'dvpt':
        model = DynamicVisualPromptTuning(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads= config['model']['heads'],
            # dim= config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
            num_prompts = config['model']['num_prompts'],
        )# .to(device)
    elif config['model']['method'] == 'evp':
        model = ExplicitVisualPrompting(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads=config['model']['heads'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            backbone=config['model']['backbone'],
            freeze_vit=config['model']['freeze_vit'],
            scale_factor=config['model']['scale_factor'],
            input_type=config['model']['input_type'],
            freq_nums=config['model']['freq_nums'],
            handcrafted_tune=config['model']['handcrafted_tune'],
            embedding_tune=config['model']['embedding_tune'],
        )# .to(device)

    elif config['model']['method'] == 'ssf':
        model = ScalingShiftingFeatures(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads=config['model']['heads'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
        )
    elif config['model']['method'] == 'melo':
        model = MedicalLoRA(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames=config['model']['frames'],
            frame_patch_size=config['model']['frame_patch_size'],
            # depth=config['model']['depth'],
            # heads=config['model']['heads'],
            # dim=config['model']['dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
        )
    elif config['model']['method'] == 'deep_vpt' or config['model']['method'] == 'shallow_vpt':
        model = PromptedVisionTransformer(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            # num_layers=config['model']['num_layers'],
            # num_heads=config['model']['num_heads'],
            # hidden_dim=config['model']['hidden_dim'],
            # mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            backbone = config['model']['backbone'],
            num_prompts = config['model']['num_prompts'],
            prompt_dropout = config['model']['prompt_dropout'],
            prompt_dim = config['model']['prompt_dim'],
            deep_prompt=config['model']['deep_prompt']
        )
    count_freeze = 0
    count_tuning = 0
    tuning_params = []
    freeze_params = []
    for name, param in model.named_parameters():
        # if 'prompt' in name or 'mlp_head' in name:
        if param.requires_grad == True:
            count_tuning += 1
            # print(name, param.shape)
            tuning_params.append(name)
        else:
            count_freeze += 1
            freeze_params.append(name)
            print(f'Freeze param: {name}')
    print(f'There are {count_tuning} trainable params.')
    print(f'including: {tuning_params}')
    print(f'There are {count_freeze} freeze params')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    print(model)
    # drop any ff and att
    # print state_dict keys
    print("Original state_dict keys:")
    for key in model.state_dict().keys():
        print(key)
    import re
    print('/n' + '-'*50 + '/n')
    # filtered_state_dict = {
    #     k: v for k, v in model.state_dict().items()
    #     if not re.search(r'\.norm\.|\.net\.|\.to_qkv\.|\.to_out\.', k, re.IGNORECASE)
    # }


    # if required gradients then filter out the keys that are required gradients
    filtered_state_dict = {
        k: v for k, v in model.state_dict().items()
        if k in tuning_params
    }

    print("Filtered state_dict keys:")
    for key in filtered_state_dict.keys():
        print(key)
    save_dir = '/mnt/e/workspace/gsoft/experiments/gaviko/test'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(filtered_state_dict, os.path.join(save_dir, 'filtered_model.pth'))



    def load_vanilla_pretrain(config):
        def pair(t):
            return t if isinstance(t, tuple) else (t, t)

        depth, heads, dim, mlp_dim = mapping_vit('vit-b16')
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

    conv_proj, num_patches = load_vanilla_pretrain(config)
    vannilla_pretrain_dicts = load_pretrain('vit-b16', num_patches, conv_proj[0].weight.shape[2], './test')
    adapters_pretain_dicts = torch.load(os.path.join(save_dir, 'filtered_model.pth'), map_location='cpu')

    #concatenate the two dictionaries
    new_dict = {**vannilla_pretrain_dicts, **adapters_pretain_dicts}
    # load the new_dict to the model
    print("New state_dict keys after merging:")
    # for key in new_dict.keys():
    #     print(key)
    model.load_state_dict(new_dict, strict=False)

    print("Model loaded successfully with filtered state_dict.")
    # print(model)
    print("model final keys:")
    # for key in model.state_dict().keys():
    #     print(key)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'model_structure_{method}.txt'), 'w') as f:
        f.write(str(model))