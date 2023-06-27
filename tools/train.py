import sys
import wandb
sys.path.append('/workspace/')

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.foldingnet2 import AutoEncoder 
from dataloader.kitti_cropped_dataloader import KITTICroppedDataloader
import torch
torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpus', type=int, default=1)
    # parser.add_argument('--max_epochs', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--k_neighbours', type=int, default=16)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--weight_decay', type=float, default=1e-4)
    # parser.add_argument('--log_dir', type=str, default='logs')
    # parser.add_argument('--log_name', type=str, default='foldingnet')
    # parser.add_argument('--save_dir', type=str, default='checkpoints')
    # args = parser.parse_args()
    wandb.login()
    kitti_cropped_dataloader = KITTICroppedDataloader(data_path='data/kitti_cropped', batch_size=1)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_top_k=1, verbose=True, monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    wandb_logger = WandbLogger(project='foldingnet', name='foldingnet')
    model = AutoEncoder()
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', logger=wandb_logger, log_every_n_steps=1, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, kitti_cropped_dataloader)
