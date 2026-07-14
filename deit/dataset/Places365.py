import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Union, List, Tuple

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Places365Dataset(Dataset):
    def __init__(self, 
                 root: str,
                 txt_file: str,
                 io_file: str,          #  IO_places365.txt 
                 cat_file: str,         #  categories_places365.txt 
                 transform: Optional[Callable] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor: float = 1.0):
        
        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor
        self.is_hier = is_hier
        self.category = category
        self.transform = transform

        self.img_path = []
        self.class_label_list = []  # Fine labels
        self.super_label_list = []  # Coarse labels

        # 1. categories_places365.txt
        #  /a/airfield 0
        name_to_fine = {}
        fine_to_name = {}
        with open(cat_file, 'r') as f:
            for line in f:
                name, fine_id = line.strip().split()
                fine_id = int(fine_id)
                name_to_fine[name] = fine_id
                fine_to_name[fine_id] = name
                
        # 2.  IO_places365.txt， fine_id -> coarse_id 
        #  /a/airfield 2 (1: Indoor, 2: Outdoor) ->  (0: Indoor, 1: Outdoor)
        fine_to_coarse = {}
        with open(io_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                name = parts[0]
                coarse_id = int(parts[1]) - 1 
                if name in name_to_fine:
                    fine_to_coarse[name_to_fine[name]] = coarse_id

        # 3.  train.txt / val.txt
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_rel_path = parts[0]
                
                if len(parts) >= 2:
                    fine_id = int(parts[1])
                else:
                    folder_name = img_rel_path.split('/')[-2]
                    fine_id = -1
                    for name, f_id in name_to_fine.items():
                        if folder_name in name:
                            fine_id = f_id
                            break
                    if fine_id == -1:
                        continue 

                coarse_id = fine_to_coarse.get(fine_id, 0)
                if img_rel_path.startswith('/'):
                    img_rel_path = img_rel_path[1:]
                    
                self.img_path.append(os.path.join(root, img_rel_path))
                self.class_label_list.append(fine_id)
                self.super_label_list.append(coarse_id)

        self.targets = self.class_label_list

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        scale_factor = self.scale_factor
        
        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)): compactness = [compactness] * len(sample)
            if not isinstance(n_segments, (list, tuple)): n_segments = [n_segments] * len(sample)
            if not isinstance(blur_ops, (list, tuple)): blur_ops = [blur_ops] * len(sample)
            if not isinstance(scale_factor, (list, tuple)): scale_factor = [scale_factor] * len(sample)

        if isinstance(sample, (list, tuple)):
            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, scale_factor):
                if blur_op is not None: samp = blur_op(samp)
                samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
                samp = (samp * 255).astype(np.uint8)
                samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                seeds = cv2.ximgproc.createSuperpixelSEEDS(
                    samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                    histogram_bins=5, double_step=False)
                seeds.iterate(samp, num_iterations=15)
                segments.append(torch.LongTensor(seeds.getLabels()))
        else:
            if blur_ops is not None: sample = blur_ops(sample)
            samp = (sample.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                histogram_bins=5, double_step=False)
            seeds.iterate(samp, num_iterations=15)
            segments = torch.LongTensor(seeds.getLabels())

        if self.is_hier:
            return sample, segments, self.class_label_list[index], self.super_label_list[index]
        else:
            if self.category == 'name':
                return sample, segments, self.class_label_list[index]    
            else:
                return sample, segments, self.super_label_list[index]