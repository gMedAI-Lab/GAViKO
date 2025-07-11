import torchio as tio
import pandas as pd
from data.dataset import CustomDataset,CustomDatasetPrediction
from torch.utils.data import DataLoader
from model.gaviko import Gaviko
from model.adaptformer import AdaptFormer
from model.bifit import BiFit
from model.dvpt import DynamicVisualPromptTuning
from model.evp import ExplicitVisualPrompting   
from model.ssf import ScalingShiftingFeatures
from model.melo import MedicalLoRA
from model.vpt import PromptedVisionTransformer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
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
    logging.info(f"Loading validation dataset from {config['data']['data_path']}")
    valid_df = pd.read_csv(config['data']['data_path'])
    valid_df = valid_df[valid_df['subset'] == 'val'].reset_index(drop=True)
    valid_ds = CustomDataset(valid_df, transforms=test_transforms,image_folder=config['data']['image_folder'])
    valid_loader = DataLoader(valid_ds, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)
    if config['model']['method'] == 'gaviko':
        model = Gaviko(**config['model'])

    elif config['model']['method'] == 'adaptformer':
        model = AdaptFormer(**config['model'])

    elif config['model']['method'] == 'bitfit':
        model = BiFit(**config['model'])
        # Freeze all parameters except for weights and head
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
        model = MedicalLoRA(**config['model'])

    elif config['model']['method'] == 'deep_vpt' or config['model']['method'] == 'shallow_vpt':
        model = PromptedVisionTransformer(**config['model'])

    model.to(device)
    logging.info(model)
    #load trained weights
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

    y_pred = []
    y_test = []
    y_pred_proba = []
    test_correct = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            y_test.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_pred_proba.extend(outputs.softmax(dim=1).cpu().numpy())

    y_test = np.array(y_test)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.array(y_pred)

    test_acc = accuracy_score(y_test, y_pred)
    test_qkv = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    logging.info(f"Test Accuracy: {test_acc}")
    logging.info(f"Test Quadratic Kappa: {test_qkv}")
    logging.info(f"Test AUC: {test_auc}")
    
    valid_df['outputs'] = y_pred.tolist()

    valid_df['mri_path'] = valid_df['mri_path'].apply(lambda x: os.path.basename(x))

    output_df = valid_df[['mri_path', 'outputs']]
    
    output_df['outputs'] = output_df['outputs'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    results_dir = config['utils']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Saving inference outputs to {results_dir}")
    version = 1
    backbone = config['model']['backbone'].replace('-', '_')
    while True:
        output_csv_name = f"{config['model']['method']}_{backbone}_eval_results_v{version}.csv"
        output_csv_path = os.path.join(results_dir, output_csv_name)
        if not os.path.exists(output_csv_path):
            break
        version += 1

    output_df.to_csv(output_csv_path, index=False)
    logging.info(f"Results saved to {output_csv_path}")

    with open(os.path.join(results_dir, f'{output_csv_name.replace(".csv", "")}_metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Quadratic Kappa: {test_qkv}\n")
        f.write(f"Test AUC: {test_auc}\n")

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
    parser.add_argument('--method', type=str, default='gaviko', choices=['gaviko', 'adaptformer', 'bitfit', 'dvpt', 'evp', 'ssf', 'melo', 'deep_vpt', 'shallow_vpt'],
                        help='Model type to use for inference')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['utils']['results_dir'] = args.results_dir if args.results_dir is not None else config['utils']['results_dir']
    config['utils']['checkpoint'] = args.checkpoint
    config['model']['method'] = args.method
    os.makedirs(config['utils']['results_dir'], exist_ok=True)
    setup_logging(log_dir=config['utils']['results_dir'])
    logging.info(f"Config: {config}")
    inference(config)

