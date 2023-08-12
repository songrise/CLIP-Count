# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import cv2
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch._six import inf
import numpy as np
import random

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def load_model_FSC(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

def load_model_FSC1(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            checkpoint1 = torch.load('./output_abnopre_dir/checkpoint-6657.pth', map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        del checkpoint1['cls_token'],checkpoint1['pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.load_state_dict(checkpoint1, strict=False)
        print("Resume checkpoint %s" % args.resume)
        

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def seed_all(seed = 42):
      # https: // www.zhihu.com/question/542479848/answer/2567626957
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def sliding_window(image, window_size = (384, 384), stride = 128):
    """
    Split an image into overlapping patches.
    Args:
        image: [3, 384, W], W >= 384
        window_size: (384, 384)
        stride: 128
    Returns:
        patches: [N, 3, 384, 384]
        intervals: [[start, end], [start, end], ...]
    """
    #right padding to make sure the image can be divided by stride
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    image = np.pad(image, ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)), 'constant')
    h, w, _ = image.shape
    assert h == 384, "FSC-147 assume image height is 384."
    patches = []
    intervals = []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])
    return np.array(patches).transpose(0,3,1,2), np.array(intervals)

# def window_composite(patches, window_size = (384, 384), stride = 128):
#     """
#     Composite patches (from sliding window) into an image.
#     for overlapping regions, average the values.
#     Args:
#         patches: [N, C, 384, 384]
#         window_size: (384, 384)
#         stride: the stride used in sliding window
#     Returns:
#         image: [1, 384, W ]
#     """
#     image = None
#     patch_h, patch_w = window_size
#     for i, patch in enumerate(patches):
#         if i == 0:
#             image = patch
#             # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
#             # cv2.imwrite(f"debug/out/image{i}.jpg", image)

#         else:
#             blend_width = patch_w - stride
#             # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
#             prev_to_blend = image[:, :, -blend_width:]
#             # cv2.imwrite(f"debug/out/prev_to_blend{i}.jpg", prev_to_blend)
#             next_to_blend = patch[:, :, :blend_width]
#             # cv2.imwrite(f"debug/out/next_to_blend{i}.jpg", next_to_blend)
#             blend = 0.5 * prev_to_blend + 0.5 * next_to_blend 
#             # cv2.imwrite(f"debug/out/blend{i}.jpg", blend)
#             image[:, :, -blend_width:] = blend
#             # cv2.imwrite(f"debug/out/image{i}.jpg", image)
#             patch_remain = patch[:, :, blend_width:]
#             #log all intermediate results
#             image = torch.cat([image, patch_remain], dim = -1)
#     return image

#soft version
def window_composite(patches, window_size = (384, 384), stride = 128):
    """
    Composite patches (from sliding window) into an image.
    for overlapping regions, average the values.
    Args:
        patches: [N, C, 384, 384]
        window_size: (384, 384)
        stride: the stride used in sliding window
    Returns:
        image: [1, 384, W ]
    """
    image = None
    patch_h, patch_w = window_size
    for i, patch in enumerate(patches):
        if i == 0:
            image = patch
            # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
            # cv2.imwrite(f"debug/out/image{i}.jpg", image)

        else:
            blend_width = patch_w - stride
            # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
            prev_to_blend = image[:, :, -blend_width:]
            # cv2.imwrite(f"debug/out/prev_to_blend{i}.jpg", prev_to_blend)
            next_to_blend = patch[:, :, :blend_width]
            # cv2.imwrite(f"debug/out/next_to_blend{i}.jpg", next_to_blend)
            blend_factor = torch.sigmoid(torch.tensor(np.linspace(-3, 3, blend_width))).to(image.device)
            blend = (1-blend_factor) * prev_to_blend + blend_factor * next_to_blend
            # cv2.imwrite(f"debug/out/blend{i}.jpg", blend)
            image[:, :, -blend_width:] = blend
            # cv2.imwrite(f"debug/out/image{i}.jpg", image)
            patch_remain = patch[:, :, blend_width:]
            #log all intermediate results
            image = torch.cat([image, patch_remain], dim = -1)
    return image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # test_img = "/workspace/songrise/CounTR/data/images_384_VarV2/7500.jpg"
    # img = cv2.imread(test_img)
    # img = np.random.randint(0,255,size=(384, 1093, 3))
    # orig_shape = img.shape

    # patches, intervals = sliding_window(img)
    patch_0 = np.zeros((3,384,384))
    patch_1 = np.ones((3,384,384))
    patches = np.array([patch_0, patch_1])
    patches = torch.from_numpy(patches)
    img_rec = window_composite(patches)
    img_rec = img_rec.permute(1,2,0).numpy()
    img_rec = img_rec*255
    # img_rec = img_rec[:, :orig_shape[1], :]
    # print(np.mean(img_rec - img))
    cv2.imwrite("test.jpg", img_rec)