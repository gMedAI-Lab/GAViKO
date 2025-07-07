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
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score


def inference(config):
    os.makedirs(config['utils']['log_dir'], exist_ok=True)
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(filename=os.path.join(config['utils']['log_dir'], f'log_{time_stamp}.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    test_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    test_df = pd.read_csv(config['data']['data_path'])

    test_ds = CustomDataset(test_df, transforms=test_transforms)
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
    #load trained weights
    model_path = config['utils']['model_path']
    if os.path.exists(model_path):
        logging.info(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.error(f"Model weights not found at {model_path}. Please check the path.")
        return

    y_pred = []
    y_test = []
    y_pred_proba = []
    test_correct = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
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
    
    test_df['outputs'] = y_pred.tolist()

    test_df['mri_path'] = test_df['mri_path'].apply(lambda x: os.path.basename(x))

    output_df = test_df[['mri_path', 'outputs']]
    
    output_df['outputs'] = output_df['outputs'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    results_dir = config['utils']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Saving inference outputs to {results_dir}")

    output_csv_path = os.path.join(results_dir, 'inference_results_details.csv')

    output_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    with open(os.path.join(results_dir, 'inference_results.txt'), 'w') as f:
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
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing MRI images')
    parser.add_argument('--results_dir', type=str, default='./',
                        help='Directory to save inference results')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory to save logs')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['data']['image_folder'] = args.image_folder
    config['data']['data_path'] = args.data_path
    config['utils']['results_dir'] = args.results_dir
    config['utils']['model_path'] = args.model_path
    os.makedirs(config['utils']['results_dir'], exist_ok=True)
    logging.info(f"Config: {config}")
    inference(config)

