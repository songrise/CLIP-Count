from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import os


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class CARPK(Dataset):
    def __init__(self, data_dir:str, split:str, subset_scale:float=1.0, resize_val:bool=True):
        """
        Parameters
        ----------
        data_dir : str, path to the data directory
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        anno_file : str, FSC-133 or FSC-147 
        """
        assert split in ['train', 'val', 'test']

        #!HARDCODED Dec 25: 
        self.data_dir = "data/CARPK/"

        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'Images')
        self.anno_path = os.path.join(self.data_dir , "Annotations")
        self.data_split_path = os.path.join(self.data_dir,'ImageSets')
        self.split = split
        self.split_file = os.path.join(self.data_split_path, split + '.txt')
        with open(self.split_file,"r") as s:
            img_names = s.readlines()
        self.idx_running_set = [x.strip() for x in img_names]
        self.gt_cnt = {}
        self.bbox = {}
        for im_name in self.idx_running_set:
            assert os.path.exists(os.path.join(self.im_dir, f"{im_name}.png"))
            assert os.path.exists(os.path.join(self.anno_path, f"{im_name}.txt"))
            with open(os.path.join(self.anno_path, f"{im_name}.txt")) as f:
                boxes = f.readlines()
                # each line is the four coordinates of a bounding box + number of cars in the bounding box
                boxes = [x.strip().split() for x in boxes]
                boxes = [[int(float(x)) for x in box][:4] for box in boxes]
                self.gt_cnt[im_name] = len(boxes)

                self.bbox[im_name] = boxes
                
        
        # resize the image height to 384, keep the aspect ratio
        self.preprocess = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
        ]
        )
            


    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_name = self.idx_running_set[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.png")
        img = Image.open(im_path)
        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        return img, gt_cnt
    
