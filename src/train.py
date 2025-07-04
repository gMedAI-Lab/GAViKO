import torchio as tio
import pandas as pd
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
from model.gaviko import VisionTransformer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from focal_loss import FocalLoss
from torch.optim.lr_scheduler import OneCycleLR

# import paht to sys

from utils.logging import CSVLogger


def train(config):
    
    os.makedirs(config['utils']['log_dir'], exist_ok=True)
    csv_logger = CSVLogger(log_dir=config['utils']['log_dir'], filename_prefix='vit_train', 
                       fields=['epoch', 'train_step_acc', 'train_step_loss', 'train_epoch_loss', 
                               'val_step_acc', 'val_step_loss', 'val_epoch_loss', 'lr', 
                               'best_epoch', 'best_val_acc', 'time_stamp', 'train_step', 'val_step'])
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(filename=os.path.join(config['utils']['log_dir'], f'log_{time_stamp}.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
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


    df = pd.read_csv(config['data']['data_path'])

    train_df = df[df['subset'] == 'train'].reset_index(drop=True)
    val_df = df[df['subset'] == 'val'].reset_index(drop=True)
    test_df = df[df['subset'] == 'test'].reset_index(drop=True)

    train_ds = CustomDataset(train_df, transforms=train_transforms, image_folder=config['data']['image_folder'])
    val_ds = CustomDataset(val_df, transforms=val_transforms, image_folder=config['data']['image_folder'])
    test_ds = CustomDataset(test_df, transforms=test_transforms, image_folder=config['data']['image_folder'])

    train_loader = DataLoader(train_ds, batch_size=config['model']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)


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

    count_freeze = 0
    count_tuning = 0
    # tuning_params = []
    # freeze_params = []
    for name, param in model.named_parameters():
        # if 'prompt' in name or 'mlp_head' in name:
        if param.requires_grad == True:
            count_tuning += 1
            # print(name, param.shape)
        else:
            count_freeze += 1
            logging.info(f'Freeze param: {name}')
    logging.info(f'Total frozen parameters: {count_freeze}')
    logging.info(f'Total parameters: {count_freeze + count_tuning}')
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


    val_acc_max = 0.0

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
            csv_logger.log({
                'epoch': current_epoch,
                'train_step_acc': train_step_acc,
                'train_step_loss': train_step_loss,
                'train_epoch_loss': train_loss,
                'val_step_acc': val_step_acc,
                'val_step_loss': val_step_loss,
                'val_epoch_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'best_epoch': current_epoch if val_acc > val_acc_max else 0,
                'best_val_acc': val_acc_max,
                'time_stamp': time_stamp,
                'train_step': train_step,
                'val_step': val_step
            })
            index += 1
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}, Current LR: {current_lr:.6f}")

        epoch_loss = running_loss / len(train_loader)
        acc = num_acc / len(train_ds)

        loss_epochs.append(epoch_loss)
        acc_epochs.append(acc)


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
                csv_logger.log({
                    'epoch': current_epoch,
                    'train_step_acc': train_step_acc,
                    'train_step_loss': train_step_loss,
                    'train_epoch_loss': train_loss,
                    'val_step_acc': val_step_acc,
                    'val_step_loss': val_step_loss,
                    'val_epoch_loss': val_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                    'best_epoch': current_epoch if val_acc > val_acc_max else 0,
                    'best_val_acc': val_acc_max,
                    'time_stamp': time_stamp,
                    'train_step': train_step,
                    'val_step': val_step
                })
                index_val += 1
    
        val_loss = running_val_loss / len(val_loader)
        val_acc = num_val_acc / len(val_ds)

        loss_val_epochs.append(val_loss)
        acc_eval_epochs.append(val_acc)


        # scheduler.step()

        current_epoch += 1
        if val_acc > val_acc_max:
            logging.info(f'Validation accuracy increased ({val_acc_max:.6f} --> {val_acc:.6f}).')
            val_acc_max = val_acc
            if val_acc_max > config['train']['save_threshold']:
                logging.info("Saving model ...")
                model_path = os.path.join(save_dir, f'best_model_epoch{current_epoch}_acc{val_acc:.4f}.pt')
                torch.save(model.state_dict(), model_path)

                save_dir = config['train']['save_dir']
                os.makedirs(save_dir, exist_ok=True)
                logging.info(f"Model saved to {model_path}")
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
            logging.info(f"There's no improvement for {epoch_since_improvement} epochs.")
            if epoch_since_improvement >= patience:
                logging.info("The training halted by early stopping criterion.")
                break
        logging.info(f"Epoch {epoch + 1}")
        logging.info(f"Loss: {epoch_loss:.4f}, Train Accuracy: {acc*100:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")

    logging.info("Training completed.")
    #save log


def main():
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser(description="Training script for Gaviko model")
    parser.add_argument('--config', type=str, default='configs/original_gaviko.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    logging.info(f"Config: {config}")
    train(config)


if __name__ == "__main__":
    main()