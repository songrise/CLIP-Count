# -*- coding : utf-8 -*-
# @FileName  : Shanghai.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Apr 30, 2023
# @Github    : https://github.com/songrise
# @Description: shanghai tech dataset
import numpy as np
from torchvision import transforms

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import os
import scipy as sp
import glob as gb
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class ShanghaiTech(Dataset):
    def __init__(self, data_dir:str, split:str, part:str, resize_val:bool=True):
        """
        Parameters
        ----------
        data_dir : str, path to the data directory
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        anno_file : str, FSC-133 or FSC-147 
        """
        assert split in ['train', 'test']
        assert part in ['A', 'B']

        #!HARDCODED Dec 25: 
        self.data_dir = "data/ShanghaiTech/part_{}/{}_data".format(part, split)

        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images')
        self.anno_path = os.path.join(self.data_dir , "ground-truth")
        # self.data_split_path = os.path.join(self.data_dir,'ImageSets')
        self.split = split
        # self.split_file = os.path.join(self.data_split_path, split + '.txt')

        # with open(self.split_file,"r") as s:
        #     img_names = s.readlines()
        self.img_paths = gb.glob(os.path.join(self.im_dir, "*.jpg"))
        self.img_names = [p.split("/")[-1].split(".")[0] for p in self.img_paths]
        self.gt_cnt = {}
        for im_name in self.img_names:

            assert os.path.exists(os.path.join(self.im_dir, f"{im_name}.jpg"))
            assert os.path.exists(os.path.join(self.anno_path, f"GT_{im_name}.mat"))
            with open(os.path.join(self.anno_path, f"GT_{im_name}.mat"), "rb") as f:
                mat = sp.io.loadmat(f)
                # the number of count is lenth of the points
                self.gt_cnt[im_name] = len(mat["image_info"][0][0][0][0][0])      
        # resize the image height to 384, keep the aspect ratio
        self.preprocess = transforms.Compose([
            transforms.Resize(384), 
            transforms.ToTensor(),
        ]
        )
            


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        im_name = self.img_names[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.jpg")
        img = Image.open(im_path)
        # if the image height larger than width, rotate it
        if img.size[0] < img.size[1]:
            img = img.rotate(90, expand=True)
        # if the image is grayscale, convert it to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        return img, gt_cnt
    

#test
if __name__ == "__main__":
    dataset = ShanghaiTech(None,split="train",part="A")
    # sample one image
    img, cnt = dataset[0]
    #save image
    img = img.permute(1,2,0).numpy()*255
    print(img.shape)
    print(cnt)
    Image.fromarray(img.astype(np.uint8)).save("test.png")
