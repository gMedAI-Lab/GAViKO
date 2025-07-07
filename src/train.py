import torchio as tio
import pandas as pd
from data.dataset import CustomDataset
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
from focal_loss import FocalLoss
from torch.optim.lr_scheduler import OneCycleLR



import wandb
# import paht to sys
from omegaconf import OmegaConf

from utils.logging import CSVLogger

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self, df):
        spatial_augment = {
            tio.RandomAffine(degrees=15, p=0.5),
            tio.RandomFlip(axes=(0), flip_probability=0.5)
        }

        intensity_augment = {
            tio.RandomNoise(): 0.25,
            tio.RandomBiasField(): 0.25,
            tio.RandomBlur(std=(0,1.5)): 0.25,
            tio.RandomMotion(): 0.25,
        }

        train_transforms = tio.Compose([
            tio.Compose(spatial_augment, p=1),
            # tio.OneOf(intensity_augment, p=0.75),
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        val_transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        test_transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])


        df = pd.read_csv(self.config['data']['data_path'])

        train_df = df[df['subset'] == 'train'].reset_index(drop=True)
        val_df = df[df['subset'] == 'val'].reset_index(drop=True)
        test_df = df[df['subset'] == 'test'].reset_index(drop=True)

        train_ds = CustomDataset(train_df, transforms=train_transforms, image_folder=self.config['data']['image_folder'])
        val_ds = CustomDataset(val_df, transforms=val_transforms, image_folder=self.config['data']['image_folder'])
        test_ds = CustomDataset(test_df, transforms=test_transforms, image_folder=self.config['data']['image_folder'])

        train_loader = DataLoader(train_ds, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=self.config['data']['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=self.config['data']['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=self.config['data']['num_workers'], pin_memory=True)
        return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def train(config):
    # Initialize WandB
    os.makedirs(config['utils']['log_dir'], exist_ok=True)
    model_name = config['model']['model_type']
    csv_logger = CSVLogger(log_dir=config['utils']['log_dir'], filename_prefix=f'{model_name}_training_log', 
                       fields=['epoch', 'train_step_acc', 'train_step_loss', 'train_epoch_loss', 
                               'val_step_acc', 'val_step_loss', 'val_epoch_loss', 'lr', 
                               'best_epoch', 'best_val_acc', 'time_stamp', 'train_step', 'val_step','train_epoch_acc', 'val_epoch_acc'])
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if config['wandb']['enable']:
        logging.info("Initializing WandB...")
        wandb.init(
            project=config['wandb']['project'],
            config=OmegaConf.to_container(config, resolve=True),
            name=config['wandb'].get('name', f"run_{time_stamp}"),
            dir=config['utils']['log_dir'],
            save_code=True,
        )
    logging.basicConfig(filename=os.path.join(config['utils']['log_dir'], f'log_{time_stamp}.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Preprocess data
    data_preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = data_preprocessor.preprocess(pd.read_csv(config['data']['data_path']))

    # Initialize model

    if config['model']['model_type'] == 'gaviko':
        model = Gaviko(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
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

    elif config['model']['model_type'] == 'adaptformer':
        model = AdaptFormer(
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
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
        )# .to(device)

    elif config['model']['model_type'] == 'bifit':
        model = BiFit(
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
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
        )
        for key, value in model.named_parameters():
            if "bias" in key:
                value.requires_grad = True
            elif "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif config['model']['model_type'] == 'dvpt':
        model = DynamicVisualPromptTuning(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            depth=config['model']['depth'],
            heads= config['model']['heads'],
            dim= config['model']['dim'],
            mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
            num_prompts = config['model']['num_prompts'],
        )# .to(device)

    elif config['model']['model_type'] == 'evp':
        model = ExplicitVisualPrompting(
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
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            pretrain_path=config['model']['pretrain_path'],
            freeze_vit=config['model']['freeze_vit'],
            scale_factor=config['model']['scale_factor'],
            input_type=config['model']['input_type'],
            freq_nums=config['model']['freq_nums'],
            handcrafted_tune=config['model']['handcrafted_tune'],
            embedding_tune=config['model']['embedding_tune'],
        )# .to(device)

    elif config['model']['model_type'] == 'ssf':
        model = ScalingShiftingFeatures(
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
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
        )
    elif config['model']['model_type'] == 'melo':
        model = MedicalLoRA(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames=config['model']['frames'],
            frame_patch_size=config['model']['frame_patch_size'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            dim=config['model']['dim'],
            mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
        )
    elif config['model']['model_type'] == 'deep_vpt' or config['model']['model_type'] == 'shallow_vpt':
        model = PromptedVisionTransformer(
            image_size=config['model']['image_size'],
            image_patch_size=config['model']['image_patch_size'],
            frames = config['model']['frames'],
            frame_patch_size = config['model']['frame_patch_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            hidden_dim=config['model']['hidden_dim'],
            mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
            channels = config['model']['channels'],
            num_classes = config['model']['num_classes'],
            freeze_vit = config['model']['freeze_vit'],
            pool = config['model']['pool'],
            pretrain_path = config['model']['pretrain_path'],
            num_prompts = config['model']['num_prompts'],
            prompt_dropout = config['model']['prompt_dropout'],
            prompt_dim = config['model']['prompt_dim'],
            deep_prompt=config['model']['deep_prompt']
        )
    model.to(device)

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
            logging.info(f'Freeze param: {name}')
    logging.info(f'There are {count_tuning} trainable params.')
    logging.info(f'including: {tuning_params}')
    logging.info(f'There are {count_freeze} freeze params')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total trainable parameters: {total_params}')
    # alpha = torch.FloatTensor([2.65, 5.39, 3.83, 7.03, 29.67]).to(device)
    criterion = FocalLoss(gamma=1.2)



    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params, lr=config['train']['lr'])

    steps_per_epoch = len(train_loader)
    num_epochs = config['train']['num_epochs']
    total_steps = steps_per_epoch * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['train']['scheduler']['max_lr'],  # learning rate cao nhất
        total_steps=total_steps,  # tổng số bước huấn luyện
        pct_start=config['train']['scheduler']['pct_start'],  # % số bước dành cho giai đoạn tăng lr (warmup)
        div_factor=config['train']['scheduler']['div_factor'],  # lr_start = max_lr / div_factor
        final_div_factor=config['train']['scheduler']['final_div_factor'],  # lr_final = lr_start / final_div_factor
        anneal_strategy=config['train']['scheduler']['anneal_strategy'],  # sử dụng cosine annealing
        three_phase=config['train']['scheduler']['three_phase']  # không dùng 3 giai đoạn (chỉ dùng 2: lên-xuống)
    )

    val_acc_max =0

    current_epoch = 0

    loss_epochs = []
    loss_val_epochs = []
    acc_epochs = []
    acc_eval_epochs = []

    patience = config['train']['patience']
    epoch_since_improvement = 0
    val_acc = 0.0
    train_acc = 0.0
    val_loss = 0.0
    train_loss = 0.0
    val_step_acc = 0.0
    train_step_acc = 0.0
    val_step_loss = 0.0
    train_step_loss = 0.0
    train_step = 0
    val_step = 0

    best_epoch = 0
    for epoch in range(num_epochs):
        num_acc = 0.0
        running_loss = 0.0
        model.train()
        index = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            m = torch.nn.Softmax(dim=-1)
            loss = criterion(m(outputs), labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            # scheduler.step(epoch + i / iters)

            running_loss += loss.item() * inputs.size(0)
            num_acc += (torch.argmax(outputs, dim = 1) == labels).sum().item()
            train_step_acc = num_acc / (len(train_ds) * (epoch + 1))
            train_step_loss = running_loss / (len(train_ds) * (epoch + 1))
            #Train step accurately
            train_step = (epoch * len(train_loader)) + index + 1
            # log at the end of batch
            # if index % config['model']['batch_size'] == 0:
            best_epoch = current_epoch if val_acc > val_acc_max else 0
            csv_logger.log({
                'epoch': current_epoch,
                'train_step_acc': train_step_acc,
                'train_step_loss': train_step_loss,
                'train_epoch_acc': train_acc,
                'train_epoch_loss': train_loss,
                'val_step_acc': val_step_acc,
                'val_step_loss': val_step_loss,
                'val_epoch_acc': val_acc,
                'val_epoch_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'best_epoch': best_epoch,
                'best_val_acc': val_acc_max,
                'time_stamp': time_stamp,
                'train_step': train_step,
                'val_step': val_step
            })
            index += 1
            if config['wandb']['enable']:
                wandb.log({
                    'train_step_acc': train_step_acc,
                    'train_step_loss': train_step_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                    'epoch': current_epoch,
                    'train_step': train_step,
                }, step=train_step)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}, Current LR: {current_lr:.6f}")

        train_loss = running_loss / len(train_loader)
        train_acc = num_acc / len(train_ds)

        loss_epochs.append(train_loss)
        acc_epochs.append(train_acc)


        # Evaluate model on epcoh
        num_val_acc = 0.0
        running_val_loss = 0.0
        final_attention_weights = []
        final_slices = []

        model.eval()
        with torch.no_grad():
            index_val = 0
            for inputs, labels in tqdm(val_loader):
                inputs  = inputs.to(device)
                labels  = labels.to(device)

                outputs = model(inputs)
                m = torch.nn.Softmax(dim=-1)
                loss = criterion(m(outputs), labels)

                running_val_loss += loss.item() * inputs.size(0)
                num_val_acc += (torch.argmax(outputs, dim = 1) == labels).sum().item()
                val_step_acc = num_val_acc / (len(val_ds) * (epoch + 1))
                val_step_loss = running_val_loss / (len(val_ds) * (epoch + 1))
                val_step = (epoch * len(val_loader)) + index_val + 1
                # if index_val % config['model']['batch_size'] == 0:
                best_epoch = current_epoch if val_acc > val_acc_max else 0
                csv_logger.log({
                    'epoch': current_epoch,
                    'train_step_acc': train_step_acc,
                    'train_step_loss': train_step_loss,
                    'train_epoch_acc': train_acc,
                    'train_epoch_loss': train_loss,
                    'val_step_acc': val_step_acc,
                    'val_step_loss': val_step_loss,
                    'val_epoch_acc': val_acc,
                    'val_epoch_loss': val_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                    'best_epoch': best_epoch,
                    'best_val_acc': val_acc_max,
                    'time_stamp': time_stamp,
                    'train_step': train_step,
                    'val_step': val_step
                })
                index_val += 1
                if config['wandb']['enable']:

                    wandb.log({
                        'val_step_acc': val_step_acc,
                        'val_step_loss': val_step_loss,
                        'epoch': current_epoch,
                        'val_step': val_step,
                    }, step=train_step)
        val_loss = running_val_loss / len(val_loader)
        val_acc = num_val_acc / len(val_ds)

        loss_val_epochs.append(val_loss)
        acc_eval_epochs.append(val_acc)


        # scheduler.step()

        current_epoch += 1
        if config['wandb']['enable']:
            wandb.log({
                'train_epoch_loss': train_loss,
                'train_epoch_acc': train_acc,
                'val_epoch_loss': val_loss,
                'val_epoch_acc': val_acc,
                'best_val_acc': val_acc_max,
            }, step=train_step)
        if val_acc > val_acc_max:
            logging.info(f'Validation accuracy increased ({val_acc_max:.6f} --> {val_acc:.6f}).')
            val_acc_max = val_acc
            if val_acc_max > config['train']['save_threshold']:
                logging.info("Saving model ...")
                model_name = config['model']['model_type']
                save_dir = os.path.join(config['train']['save_dir'],'experiments', model_name)
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f'{model_name}_best_model_epoch{current_epoch}_acc{val_acc:.4f}.pt')
                torch.save(model.state_dict(), model_path)

                logging.info(f"Model saved to {model_path}")
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
            logging.info(f"There's no improvement for {epoch_since_improvement} epochs.")
            if epoch_since_improvement >= patience:
                logging.info("The training halted by early stopping criterion.")
                break
        logging.info(f"Epoch {epoch + 1}")
        logging.info(f"Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")

    logging.info("Training completed.")
    #save log
    # Update WandB summary
    # wandb.run.summary['best_val_acc'] = val_acc_max
    # wandb.run.summary['best_epoch'] = best_epoch

    # Finish the run
    wandb.finish()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Training script for Gaviko model")
    parser.add_argument('--config', type=str, default='configs/original_gaviko.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model', type=str, default='gaviko',
                        choices=['gaviko', 'adaptformer', 'bifit', 'dvpt', 'evp', 'ssf', 'melo', 'deep_vpt','shallow_vpt'],
                        help='Model to train: gaviko, adaptformer, bifit, dvpt, evp, ssf, melo, deep_vpt, shallow_vpt')
    parser.add_argument('--wandb',default=None, help='Enable WandB logging')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['wandb']['enable'] = args.wandb if args.wandb is not None else config['wandb']['enable']
    config['model']['model_type'] = args.model 
    if config['model']['model_type'] == 'deep_vpt':
        config['model']['deep_prompt'] = True
    elif config['model']['model_type'] == 'shallow_vpt':
        config['model']['deep_prompt'] = False
    logging.info(f"Config: {config}")
    train(config)


if __name__ == "__main__":
    main()