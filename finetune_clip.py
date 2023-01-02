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
from typing import Optional, List, Dict, Any, Union, Tuple
import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import  FSC147
from models import models_mae_cross, models_clip, models_clip_regress
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.lite import LightningLite
import einops
import cv2 
import gradio as gr
from torchvision.transforms import Resize

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument("--mode",type = str, default = "train", choices = ["train","test", "app"], help = "train or test")
    parser.add_argument("--exp_name",type = str, default = "exp", help = "experiment name")
    parser.add_argument('--batch_size', default=26, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--backbone', default="b16", choices=["b16", "b32"], 
                    type=str, help = "backbone of clip")
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
    parser.add_argument('--ckpt', default=None, type = str,
                        help='path of resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)
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

    # prompt learning
    parser.add_argument('--use_coop', action='store_true',
                        help='whether to perform context learning for text prompts.')
    parser.add_argument('--use_vpt', action='store_true',
                        help='whether to perform visual prompt learning.')

    #loss relate
    parser.add_argument("--use_contrast", action='store_true', help = "whether to use contrasitive loss")
    return parser


class Model(LightningModule):
    def __init__(self, args, all_classes:List[str] = None):
        super().__init__()
        self.args = args
        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)
        self.all_classes = all_classes

        self.save_hyperparameters(args)
        self.model = models_clip.CLIPCount(
                        use_coop=self.args.use_coop, 
                        use_vpt=self.args.use_vpt,
                        # backbone = self.args.backbone
                        )
        self.loss = F.mse_loss
        # self.loss = FocalLoss()
        # self.loss = F.binary_cross_entropy

    def training_step(self, batch, batch_idx):


        samples, gt_density, boxes, m_flag, prompt_gt = batch

        output = self.model(samples, prompt_gt)
        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384,384])
        masks = np.tile(mask,(output.shape[0],1))
        masks = masks.reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)
        #todo test focal loss
        loss = self.loss(output, gt_density)
        # loss for negative prompt

        # todo the loss should take the number into account otherwise easy for degenerated sol
        loss = (loss * masks / (384*384)).sum() / output.shape[0]
        if self.args.use_contrast:
            prompt_neg = self.gen_negative_prompt(prompt_gt)
            output_neg = self.model(samples, prompt_neg)
            neg_density = torch.zeros_like(gt_density) # for negative prompt, density is all zero
            loss_neg = self.loss(output_neg, neg_density)
            loss = loss + loss_neg * 0.5
        # if args.use_sparsity:
            # sparsity_loss = -torch.log(torch.exp(-torch.abs(output))+torch.exp(-torch.abs(1.-output)))
            # sparsity_loss = torch.mean(sparsity_loss + 0.31326165795326233)
        self.log('train_loss', loss)

        # Update information of MAE and RMSE
        batch_mae = 0
        batch_rmse = 0
        gt_sum = 0
        #TODO Dec 27: parallelize this
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/60).item()
            gt_cnt = torch.sum(gt_density[i]/60).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            gt_sum += gt_cnt
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse)
        # loss = loss / gt_sum
        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)
    
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        samples, gt_density, boxes, m_flag, prompt = batch
        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.


        output = self.model(samples, prompt)
        # output = einops.rearrange(output, 'b h w -> b (h w)')
        # gt_density = einops.rearrange(gt_density, 'b h w -> b (h w)')
        #! Dec 24: add sigmoid to output here
        # output = F.sigmoid(output)

        
        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/60).item()
            gt_cnt = torch.sum(gt_density[i]/60).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae.append(cnt_err)
            batch_rmse.append(cnt_err ** 2)
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)


        #log the image
        img_log = samples[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2,0,1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255*gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2,0,1))


        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max() #normalize
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = 0.33 * img_log + 0.67 * gt_density_log

        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb, "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}
    
    def validation_epoch_end(self, outputs):
        #TODO Jan 02: has bug in calculate rmse
        all_mae = []
        all_rmse = []
        self.print(len(all_rmse))
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)

        # log the image
        idx = random.randint(0, len(outputs)-1)
        img = outputs[idx]["img"]
        pred = outputs[idx]["pred"]
        gt = outputs[idx]["gt"]
        heatmap_pred = outputs[idx]["heatmap_pred"]
        heatmap_gt = outputs[idx]["heatmap_gt"]
        prompt = outputs[idx]["prompt"]
        pred_cnts = outputs[idx]["pred_cnts"]
        gt_cnts = outputs[idx]["gt_cnts"]
        pred_gt = "pred: {:.2f} gt: {:.2f}".format(pred_cnts[0], gt_cnts[0])
        self.logger.experiment.add_image("val_img", img, self.current_epoch)
        self.logger.experiment.add_image("density_pred", pred, self.current_epoch)
        self.logger.experiment.add_image("density_gt", gt, self.current_epoch)
        self.logger.experiment.add_image("overlay_pred", heatmap_pred, self.current_epoch)
        self.logger.experiment.add_image("overlay_gt", heatmap_gt, self.current_epoch)
        self.logger.experiment.add_text("prompt", prompt, self.current_epoch)
        self.logger.experiment.add_text("count", pred_gt, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
                # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        samples, gt_density, boxes, m_flag, prompt = batch

        #! Jan 02: test text
        prompt = self.gen_negative_prompt(prompt)

        #! Jan 02: revise
        samples = Resize((384, 384))(samples)

        output = self.model(samples, prompt)

        
        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/60).item()
            gt_cnt = torch.sum(gt_density[i]/60).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae.append(cnt_err)
            batch_rmse.append(cnt_err ** 2)
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)
 

        #log the image
        img_log = samples[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2,0,1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255*gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2,0,1))


        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max() #normalize
        heatmap_pred = img_log #! Jan 02: temp for debugging
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = img_log #! Jan 02: temp for debugging
        heatmap_gt = 0.33 * img_log + 0.67 * gt_density_log


        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb, "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}
    
    def test_epoch_end(self, outputs):
        all_mae = []
        all_rmse = []
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        test_mae = np.mean(all_mae)
        test_rmse = np.sqrt(np.mean(all_rmse))
        self.log('test_mae', test_mae)
        self.log('test_rmse', test_rmse)

    def forward(self, img, prompt):
        """
        img: (1, 3, H, W)
        prompt: List[str]
        """
        return self.model(img, prompt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95)
        )

        # schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def gen_negative_prompt(self,prompt_gt:List[str]):
        prompts = []
        for i in range(len(prompt_gt)):
            neg_prompt = prompt_gt[i]
            while neg_prompt == prompt_gt[i]:
                neg_prompt = random.choice(self.all_classes)
                # neg_prompt = random.choice(["dog","grapes", "airplane", "birds", "bottle caps", "objects","people","potted plants","shoes","table"])
                # neg_prompt = "objects"
            prompts.append(neg_prompt)

        return prompts
  

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed_everything(1)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    

    dataset_train = FSC147(args.data_path, split = "train", subset_scale=1)
    all_classes_train = dataset_train.all_classes
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

    dataset_test = FSC147(args.data_path, split = "test")
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    test_dataloader =  torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    #seed everything

    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=2, mode='min',  filename='{epoch}-{val_mae:.2f}',)
    model = Model(args,all_classes=all_classes_train)
    model = Model.load_from_checkpoint("/root/autodl-tmp/CLIPCount/lightning_logs/vitB16enc_fix/version_0/checkpoints/epoch=92-val_mae=20.93.ckpt")
    # prof = pl.profilers.AdvancedProfiler(dirpath = ".",filename="perf_logs")
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name)
    trainer = Trainer(
        accelerator="gpu", 
        # log_every_n_steps=50, 
        callbacks=[save_callback],
        accumulate_grad_batches = 1, 
        precision=16, 
        max_epochs=args.epochs,
        logger=logger,
        check_val_every_n_epoch=3
    )#, profiler=prof,max_epochs=1)
    if args.mode == "train":
        #overfit
        # trainer.fit(model, train_dataloader, train_dataloader) #!HARDCODED Dec 28: 
        #normal
        trainer.fit(model, train_dataloader, val_dataloader)
        #test
        # trainer.test(model, test_dataloader)
    elif args.mode == "test":
        model.eval()
        trainer.test(model, val_dataloader)

    elif args.mode == "app":
        def infer(img, prompt):
            model.eval()
            model.model = model.model.cuda()
            with torch.no_grad():

                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                img = img.float()/255
                prompt = [prompt]
                with torch.cuda.amp.autocast():
                    pred = model.forward(img, prompt)
                pred_cnt = torch.sum(pred/60).item()
                pred_density = pred[0].detach().cpu().numpy()
                pred_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
            return pred_rgb, pred_cnt
        demo = gr.Interface(
            fn=infer,
            inputs=[
                gr.inputs.Image(shape=(224, 224), label="Image"),
                gr.inputs.Textbox(lines=1, label="Prompt"),
            ],
            outputs= ["image", "number"],
            interpretation="default",
            title="CLIPCount",
            description="Counting with CLIP",
            
        )
        demo.launch(share=True)