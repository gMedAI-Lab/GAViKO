import torchio as tio
import pandas as pd
from data.dataset import CustomDataset,CustomDatasetPrediction
from torch.utils.data import DataLoader
from model.adaptformer import AdaptFormer
from model.vision_transformer import VisionTransformer
from model.dvpt import DynamicVisualPromptTuning
from model.evp import ExplicitVisualPrompting
from model.ssf import ScalingShiftingFeatures
from model.melo import MeLO
from model.vpt import PromptedVisionTransformer
from model.gaviko import Gaviko
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from utils.load_pretrained import load_vanilla_pretrain_with_adapters
from utils.logging import setup_logging

def inference(config):
    os.makedirs(config['utils']['log_dir'], exist_ok=True)
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    # logging.basicConfig(filename=os.path.join(config['utils']['log_dir'], f'log_{time_stamp}.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Phase: {config['utils']['phase']}")

    test_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    test_df = generate_csv(config['data']['image_folder'])

    test_ds = CustomDatasetPrediction(test_df, transforms=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)

    if config['model']['method'] == 'gaviko':
        model = Gaviko(**config['model'])

    elif config['model']['method'] == 'linear':
        model = VisionTransformer(**config['model'])
        # Freeze all parameters except for weights and head
        for key, value in model.named_parameters():
            if "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False
    
    elif config['model']['method'] == 'fft':
        model = VisionTransformer(**config['model'])

    elif config['model']['method'] == 'adaptformer':
        model = AdaptFormer(**config['model'])

    elif config['model']['method'] == 'bitfit':
        model = VisionTransformer(**config['model'])
        for key, value in model.named_parameters():
            if "bias" in key:
                value.requires_grad = True
            elif "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False

    elif config['model']['method'] == 'dvpt':
        model = DynamicVisualPromptTuning(**config['model'])

    elif config['model']['method'] == 'evp':
        model = ExplicitVisualPrompting(**config['model'])

    elif config['model']['method'] == 'ssf':
        model = ScalingShiftingFeatures(**config['model'])

    elif config['model']['method'] == 'melo':
        vit_model = VisionTransformer(**config['model'])
        model = MeLO(vit=vit_model, **config['model'])

    elif config['model']['method'] == 'deep_vpt' or config['model']['method'] == 'shallow_vpt':
        model = PromptedVisionTransformer(**config['model'])

    model.to(device)
    # Load trained weights
    logging.info(f"Model type: {config['model']['method']}")

    if config['utils']['checkpoint']:
        checkpoint_path = config['utils']['checkpoint']
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading model weights from {checkpoint_path}")
            model_dict = load_vanilla_pretrain_with_adapters(config['model']['backbone'],config,checkpoint_path)
            model.load_state_dict(model_dict, strict=False)
            
        else:
            raise FileNotFoundError(f"Model weights not found at {checkpoint_path}. Please check the path.")
    else:
        logging.info(f"Model path is not provided. {config['model']['method']} weights are initialized randomly.")
    
    logging.info(model)

    model.eval()

    all_outputs = []

    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Running Inference"):
            inputs = inputs.to(device)  


            outputs = model(inputs)  

            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy() 

            all_outputs.append(predicted_classes)
    all_outputs = np.concatenate(all_outputs, axis=0)  
    logging.info(f"Final outputs shape: {all_outputs.shape}")

    test_df['outputs'] = all_outputs.tolist()

    test_df['mri_path'] = test_df['mri_path'].apply(lambda x: os.path.basename(x))

    output_df = test_df[['mri_path', 'outputs']]

    output_df['outputs'] = output_df['outputs'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    results_dir = config['utils']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    method = config['model']['method']
    backbone = config['model']['backbone'].replace('-', '_') 
    # find in dir if already exists, increment the version number
    version = 1
    while True:
        output_csv_name = f"{method}_{backbone}_inference_results_v{version}.csv"
        output_csv_path = os.path.join(results_dir, output_csv_name)
        if not os.path.exists(output_csv_path):
            break
        version += 1
    logging.info(f"Saving results to {output_csv_path}")
    output_df.to_csv(output_csv_path, index=False)
    logging.info(f"Results saved to {output_csv_path}")

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
    parser.add_argument('--results_dir', type=str, default='./outputs',
                        help='Directory to save inference results')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to the trained model weights')
    parser.add_argument('--method', type=str, default='gaviko', choices=['gaviko', 'fft', 'linear', 'adaptformer', 'bitfit', 'dvpt', 'evp', 'ssf', 'melo', 'deep_vpt', 'shallow_vpt'],
                        help='Type of model to use (default: gaviko)')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['model']['method'] = args.method 
    if config['model']['method'] == 'deep_vpt':
        config['model']['deep_prompt'] = True
    elif config['model']['method'] == 'shallow_vpt':
        config['model']['deep_prompt'] = False
    # config['data']['image_folder'] = args.image_folder
    config['utils']['results_dir'] = args.results_dir if args.results_dir is not None else config['utils']['results_dir']
    config['utils']['checkpoint'] = args.checkpoint
    os.makedirs(config['utils']['results_dir'], exist_ok=True)
    setup_logging(log_dir=config['utils']['results_dir'])
    logging.info(f"Config: {config}")
    inference(config)