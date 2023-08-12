# -*- coding : utf-8 -*-
# @FileName  : run.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Aug 13, 2023
# @Github    : https://github.com/songrise
# @Description: script to train and test CLIP-Count
#supress torchvision warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import os
import random
from pathlib import Path
import math
from PIL import Image
from models.contrastive_loss import ContrastiveLoss
import torch
import torch.nn.functional as F
from typing import List, Dict, Any


import util.misc as misc
from util.FSC147 import  FSC147
from util.CARPK import CARPK
from util.ShanghaiTech import ShanghaiTech
import util
from models import  clip_count
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
import einops
import cv2 
import gradio as gr
import torchvision.transforms.functional as TF
from util.constant import SCALE_FACTOR

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-Count', add_help=False)
    parser.add_argument("--mode",type = str, default = "train", choices = ["train", "test", "app"], help = "train or test or an interactive application")
    parser.add_argument("--exp_name",type = str, default = "exp", help = "experiment name")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Model parameters

    parser.add_argument('--backbone', default="b16", choices=["b16", "b32", "l14"], 
                    type=str, help = "backbone of clip")
    parser.add_argument('--decoder_depth', default=4, type=int, help='Number of FIM layers')
    parser.add_argument('--decoder_head', default=8, type=int, help='Number of attention heads for FIM')

    parser.add_argument('--use_mixed_fim', default=True, type = misc.str2bool, help = "whether to use hierarchical patch-text interaction")
    parser.add_argument('--unfreeze_vit', default=False, type = misc.str2bool, help = "whether to unfreeze CLIP vit i.e., finetune CLIP")
    parser.add_argument('--use_fim', default=False, type = misc.str2bool, help = "whether to use naive interaction")
    
    #contrastive loss related
    parser.add_argument('--use_coop',  default=True, type = misc.str2bool,
                        help='whether to perform context learning for text prompts.')
    parser.add_argument('--coop_width', default = 2, type = int, help = "width of context (how many token to be learned)")
    parser.add_argument('--coop_require_grad', default = False, type = misc.str2bool, help = "whether to require grad for context learning")
    parser.add_argument('--use_vpt', default=True, type = misc.str2bool,
                        help='whether to perform visual prompt learning.')
    parser.add_argument('--vpt_width', default = 20, type = int, help = "width of visual prompt (how many token each layer)")
    parser.add_argument('--vpt_depth', default = 10, type = int, help = "depth of visual prompt (how many layer)")

    parser.add_argument("--use_contrast", default=True, type = misc.str2bool, help = "whether to use contrasitive loss")
    parser.add_argument("--w_contrast", default = 1.0, type = float, help = "weight of contrastive loss")
    parser.add_argument("--noise_text_ratio", default = 0.0, type = float, help = "ratio of noise text")
    parser.add_argument('--normalize_contrast',default=False, type = misc.str2bool, help = "whether to normalize contrastive loss")
    parser.add_argument('--contrast_pos', default = "pre", choices = ["pre", "post"], type = str, help = "Use contrastive loss before or after the interaction")
    parser.add_argument('--contrast_pre_epoch', default = 20, type = int, help = "how many epoch to use contrastive pretraining")
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')


    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_type', default="FSC", type = str, choices=["FSC","CARPK", "COCO", "ShanghaiTech"])

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=1, type=int)


    parser.add_argument('--ckpt', default=None, type = str,
                        help='path of resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # log related
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--log_test_img', default=False, type=bool, help="whehter to log overlaied density map when validation and testing.")
    parser.add_argument('--dont_log', action='store_true', help='do not log to tensorboard')
    parser.add_argument('--val_freq', default=1, type=int, help='check validation every val_freq epochs')




    #log setup
    parser.add_argument('--exp_note', default = "", type = str, help = "experiment note")
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
        self.model = clip_count.CLIPCount(
                        fim_depth=self.args.decoder_depth,
                        fim_num_heads=self.args.decoder_head,
                        use_coop=self.args.use_coop, 
                        use_vpt=self.args.use_vpt,
                        coop_width=self.args.coop_width,
                        vpt_width=self.args.vpt_width,
                        vpt_depth= self.args.vpt_depth,
                        backbone = self.args.backbone,
                        use_fim = self.args.use_fim,
                        use_mixed_fim = self.args.use_mixed_fim,
                        unfreeze_vit = self.args.unfreeze_vit,
                        )
        self.loss = F.mse_loss
        self.contrastive_loss = ContrastiveLoss(0.07,self.args.noise_text_ratio, self.args.normalize_contrast)
        self.neg_prompt_embed = None

    def training_step(self, batch, batch_idx):

        samples, gt_density, boxes, m_flag, prompt_gt, prompt_add = batch


        output, extra_out = self.model(samples, prompt_gt, return_extra=True, coop_require_grad =  True)

        if not self.args.use_contrast:
            prompt_gt = [f"a photo of {p}" for p in prompt_gt]
        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384,384])
        masks = np.tile(mask,(output.shape[0],1))
        masks = masks.reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)
        loss = self.loss(output, gt_density)

        loss = (loss * masks / (384*384)).sum() / output.shape[0]
        if self.args.use_contrast and self.current_epoch <= self.args.contrast_pre_epoch:
            text_embedding = extra_out['text_embedding'] # [B,1, 512]
            if self.args.contrast_pos == "pre":
                patch_embedding = extra_out['patch_embedding_contrast'] # [B, 196, 512]
            elif self.args.contrast_pos == "post":
                patch_embedding = extra_out['pixel_text_matching_map']
            img_embedding = extra_out['x_cls'] # [B, 1, 512]
            contrast_loss = self.contrastive_loss(patch_embedding, img_embedding, text_embedding, self.neg_prompt_embed,  gt_density.detach().clone())
            loss = args.w_contrast * contrast_loss
            self.log('train_loss_contrast', contrast_loss)


        self.log('train_loss', loss)


        # Update information of MAE and RMSE
        batch_mae = 0

        batch_rmse = 0
        gt_sum = 0
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/SCALE_FACTOR).item()
            gt_cnt = torch.sum(gt_density[i]/SCALE_FACTOR).item()
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
        samples, gt_density, _, _, prompt, _ = batch
        if not self.args.use_contrast:
            prompt = [f"a photo of {p}" for p in prompt]

        output = self.model(samples, prompt)

        
        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/SCALE_FACTOR).item() # SCALE_FACTOR is the scaling factor as CounTR uses
            gt_cnt = torch.sum(gt_density[i]/SCALE_FACTOR).item()
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
        all_mae = []
        all_rmse = []

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
        if self.args.dataset_type=='FSC' or self.args.dataset_type == "COCO":
            image, gt_density, boxes, m_flag, prompt = batch
        elif self.args.dataset_type == "CARPK":
            image, gt_cnt = batch
            gt_cnt = gt_cnt.item()
            prompt = ["car" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3]) 
        elif self.args.dataset_type == "ShanghaiTech":
            image, gt_cnt = batch
            gt_cnt = gt_cnt.item()
            prompt = ["people" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3]) 


        assert image.shape[0] == 1 , "only support inference one image at a time"
        raw_h, raw_w = image.shape[2:]

        patches, _ = misc.sliding_window(image,stride=128)
        #covert to batch
        patches = torch.from_numpy(patches).float().to(self.device)
        prompt = np.repeat(prompt, patches.shape[0], axis=0)
        output = self.model(patches, prompt)
        output.unsqueeze_(1)
        output = misc.window_composite(output, stride=128)
        output = output.squeeze(1)
        #crop to original width
        output = output[:, :, :raw_w]

        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []

        pred_cnt = torch.sum(output[0]/SCALE_FACTOR).item()
        if self.args.dataset_type == "FSC" or self.args.dataset_type == "COCO":
            gt_cnt = torch.sum(gt_density[0]/SCALE_FACTOR).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        batch_mae.append(cnt_err)
        batch_rmse.append(cnt_err ** 2)
        pred_cnts.append(pred_cnt)
        gt_cnts.append(gt_cnt)
 

        #log the image
        img_log = image[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2,0,1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255*gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2,0,1))


        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max() #normalize
        heatmap_pred = img_log 
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = img_log 

        # log qualitative results
        if self.args.log_test_img:
            if cnt_err < 5:
                #log density
                log_dir = "out/good_density/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)
                img = Image.fromarray(np.uint8(pred_density_write))
                img.save(log_dir + name)

                log_dir = "out/good_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                #log overlay
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log,(1,2,0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255*img_write))
                img.save(log_dir + name)

            if cnt_err > 100:
                #save image, overlaied
                #log density
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)

                log_dir = "debug/bad_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "bad_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log,(1,2,0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255*img_write))
                img.save(log_dir + name)

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
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # delete frozen clip parameters
        if not self.args.unfreeze_vit :
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("model.clip") or k.startswith("model.img_encoder.clip") or k.startswith("model.text_encoder.clip") or k.startswith("model.img_encoder.vit"):
                    del checkpoint["state_dict"][k]

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        self.args = args

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    

    dataset_train = FSC147(split = "train")
    all_classes_train = dataset_train.all_classes
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    # the val set for training.
    dataset_val = FSC147( split = "val")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader =  torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )



    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=4, mode='min',  filename='{epoch}-{val_mae:.2f}')
    model = Model(args,all_classes=all_classes_train)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name)
    trainer = Trainer(
        accelerator="gpu", 
        callbacks=[save_callback],
        accumulate_grad_batches = args.accum_iter, 
        precision=16, 
        max_epochs=args.epochs+args.contrast_pre_epoch,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
    )
    if args.mode == "train":
        if args.ckpt is not None:
            model = Model.load_from_checkpoint(args.ckpt, strict=False)

        trainer.fit(model, train_dataloader, val_dataloader)
    elif args.mode == "test":
        if args.dataset_type == "FSC":
            dataset_val = FSC147(split = "val", resize_val=False)
            dataset_test = FSC147(split = "test")
        elif args.dataset_type == "COCO":
            dataset_val = FSC147(split = "val_coco", resize_val=False)
            dataset_test = FSC147(split = "test_coco")

        elif args.dataset_type == "CARPK":
            dataset_val = dataset_test = CARPK(None, split="test")
        elif args.dataset_type == "ShanghaiTech":
            dataset_val = dataset_test = ShanghaiTech(None, split="test", part = "B")


        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # when inference, batch size is always 1
        val_dataloader =  torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt,strict=False)
        model.overwrite_args(args)
        model.eval()
        if args.dataset_type == "FSC" or args.dataset_type == "COCO": #CARPK and ShanghaiTech do not have val set
            print("====Metric on val set====")
            trainer.test(model, val_dataloader)
        print("====Metric on test set====")
        trainer.test(model, test_dataloader)



    elif args.mode == "app":
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt,strict=False)
        model.eval()
        def infer(img, prompt):
            model.eval()
            model.model = model.model.cuda()
            with torch.no_grad():
                # reshape height to 384, keep aspect ratio
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                img = TF.resize(img, (384))
                
                img = img.float()/255.
                img = torch.clamp(img, 0, 1)
                prompt = [prompt]
                with torch.cuda.amp.autocast():
                    raw_h, raw_w = img.shape[2:]
                    patches, _ = misc.sliding_window(img,stride=128)
                    #covert to batch
                    patches = torch.from_numpy(patches).float().to(img.device)
                    prompt = np.repeat(prompt, patches.shape[0], axis=0)
                    output = model.forward(patches, prompt)
                    output.unsqueeze_(1)
                    output = misc.window_composite(output, stride=128)
                    output = output.squeeze(1)
                    #crop to original width
                    output = output[:, :, :raw_w]
                pred_cnt = torch.sum(output[0]/SCALE_FACTOR).item()
                pred_density = output[0].detach().cpu().numpy()
                # normalize
                pred_density = pred_density/pred_density.max()
                pred_density_write = 1. - pred_density
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)
                pred_density_write = pred_density_write/255.
                # pred_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
                img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                
                heatmap_pred = 0.33 * img + 0.67 * pred_density_write
                heatmap_pred = heatmap_pred/heatmap_pred.max()
            return heatmap_pred, pred_cnt
        demo = gr.Interface(
            fn=infer,
            inputs=[
                # height = 384, keep aspect ratio
                gr.inputs.Image(label="Image"),
                gr.inputs.Textbox(lines=1, label="Prompt (What would you like to count)"),
            ],
            outputs= ["image", "number"],
            interpretation="default",
            title="CLIP-Count",
            description="A unified counting model to count them all.",
            
        )
        demo.launch(share=True)