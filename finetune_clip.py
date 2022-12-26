import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import  FSC147
from models import models_mae_cross, models_clip
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.lite import LightningLite



os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=26, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--resume', default='./output_pre_4_dir/checkpoint-300.pth',
    #                    help='resume from checkpoint')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # tricks
    parser.add_argument('--learn_prompt', action='store_true',
                        help='whether to perform prompt learning.')
    return parser


class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.save_hyperparameters(args)
        self.model = models_clip.CLIPCount(learn_context=self.args.learn_prompt)
        self.loss = F.mse_loss
        # self.loss = FocalLoss()
        # self.loss = F.binary_cross_entropy

    def training_step(self, batch, batch_idx):


        samples, gt_density, boxes, m_flag, prompt = batch
        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        flag = 0
        for i in range(m_flag.shape[0]):
            flag += m_flag[i].item()
        if flag == 0:
            shot_num = random.randint(0,3)
        else:
            shot_num = random.randint(1,3)

        # output = self.model(samples,boxes,shot_num)
        output = self.model(samples, prompt)

        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384,384])
        masks = np.tile(mask,(output.shape[0],1))
        masks = masks.reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)
        #todo test focal loss
        # output = F.sigmoid(output) #! Dec 24: add sigmoid to output here
        loss = self.loss(output, gt_density)
        # loss = (loss * masks / (384*384)).sum() / output.shape[0]
        self.log('train_loss', loss)

        # Update information of MAE and RMSE
        batch_mae = 0
        batch_rmse = 0
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/60).item()
            gt_cnt = torch.sum(gt_density[i]/60).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse)
        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        samples, gt_density, boxes, m_flag, prompt = batch
        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        flag = 0
        for i in range(m_flag.shape[0]):
            flag += m_flag[i].item()
        if flag == 0:
            shot_num = random.randint(0,3)
        else:
            shot_num = random.randint(1,3)

        # output = self.model(samples,boxes,shot_num)
        output = self.model(samples, prompt)
        #! Dec 24: add sigmoid to output here
        # output = F.sigmoid(output)

        
        # Update information of MAE and RMSE
        batch_mae = 0
        batch_rmse = 0
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/60).item()
            gt_cnt = torch.sum(gt_density[i]/60).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse)
        self.log('val_mae', batch_mae)
        self.log('val_rmse', batch_rmse)
        # self.print('val_mae = ', batch_mae)
        # self.print('val_rmse = ', batch_rmse)
        return batch_mae, batch_rmse
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95)
        )
        return optimizer

  

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed_everything(1)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    

    dataset_train = FSC147(args.data_path, split = "train")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    dataset_val = FSC147(args.data_path, split = "val")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader =  torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    #seed everything

    save_callback = pl.callbacks.ModelCheckpoint()
    model = Model(args)
    # model = Model.load_from_checkpoint("/root/autodl-tmp/CounTR/lightning_logs/version_1/checkpoints/epoch=8-step=324.ckpt")
    # prof = pl.profilers.AdvancedProfiler(dirpath = ".",filename="perf_logs")
    trainer = Trainer(accelerator="gpu", log_every_n_steps=50, accumulate_grad_batches = 8, precision=16)#, profiler=prof,max_epochs=1)
    trainer.fit(model, train_dataloader, val_dataloader)
