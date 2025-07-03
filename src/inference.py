import torchio as tio
import pandas as pd
from data.dataset import CustomDataset,CustomDatasetPrediction
from torch.utils.data import DataLoader
from model.gaviko import VisionTransformer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np


def inference(config):
    os.makedirs(config['utils']['log_dir'], exist_ok=True)
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(filename=os.path.join(config['utils']['log_dir'], f'log_{time_stamp}.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    test_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    test_df = generate_csv(config['data']['image_folder'])

    test_ds = CustomDatasetPrediction(test_df, transforms=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)
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
        pretrain_path = config['model']['pretrain_path'],
        num_prompts=config['model']['num_prompts'],
        prompt_latent_dim=config['model']['prompt_latent_dim'],
        local_dim=config['model']['local_dim'],
        local_k= tuple(config['model']['local_k']),
        DHW=tuple(config['model']['DHW']),
        share_factor=config['model']['share_factor']
    )
    model.to(device)

    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())   
    all_outputs = np.concatenate(all_outputs, axis=0)
    logging.info(f"Shape of all_outputs: {all_outputs.shape}")
    output_df = pd.DataFrame(all_outputs, columns=[f'output_{i}' for i in range(all_outputs.shape[1])])
    output_csv_path = os.path.join(config['utils']['results_dir'], 'inference_outputs.csv')
    output_df.to_csv(output_csv_path, index=False)
    logging.info(f"Inference outputs saved to {output_csv_path}")
def generate_csv(image_folder):
    """
    Generates a CSV file with image paths and labels.
    :param image_folder: Path to the folder containing images.
    :param output_csv_path: Path where the output CSV will be saved.
    """
    image_paths = []
    # labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.npz'):
            image_paths.append(os.path.join(image_folder, filename))
            # label = filename.split('_')[-1].split('.')[0]  # Extracting label from filename
            # labels.append(label)
    
    df = pd.DataFrame({'mri_path': image_paths})
    # df.to_csv(output_csv_path, index=False)
    return df
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for Gaviko model")
    parser.add_argument('--config', type=str, default='/workspace/train_deep_prompt/configs/original_gaviko.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing MRI images')
    parser.add_argument('--results_dir', type=str, default='./',
                        help='Directory to save inference results')
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['data']['image_folder'] = args.image_folder
    config['utils']['results_dir'] = args.results_dir
    os.makedirs(config['utils']['results_dir'], exist_ok=True)
    logging.info(f"Config: {config}")
    inference(config)

