import logging
from src.model.gaviko import VisionTransformer
import torch


def main():
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser(description="Training script for Gaviko model")
    parser.add_argument('--config', type=str, default='/mnt/e/workspace/gsoft/GAViKO_origin/src/configs/original_gaviko.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    logging.info(f"Config: {config}")
    model = VisionTransformer(
        image_size=config['model']['image_size'],
        image_patch_size=config['model']['image_patch_size'],
        frames = config['model']['frames'],
        frame_patch_size = config['model']['frame_patch_size'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim=config['model']['dim'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['model']['dropout'],
        emb_dropout=config['model']['emb_dropout'],
        attn_drop=config['model']['attn_drop'],
        proj_drop=config['model']['proj_drop'],
        channels=config['model']['channels'],
        num_classes=config['model']['num_classes'],
        freeze_vit = config['model']['freeze_vit'],
        pool = config['model']['pool'],
        pretrain_path = '/mnt/e/workspace/gsoft/vit-base-patch16-224-in21k.pth',
        num_prompts=config['model']['num_prompts'],
        prompt_latent_dim=config['model']['prompt_latent_dim'],
        local_dim=config['model']['local_dim'],
        local_k= tuple(config['model']['local_k']),
        DHW=tuple(config['model']['DHW']),
        share_factor=config['model']['share_factor']
    )

    print(model)

if __name__ == "__main__":
    main()